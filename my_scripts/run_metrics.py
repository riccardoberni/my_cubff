# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_metrics.py
==============

Utility per raccogliere e visualizzare le metriche di una qualsiasi run di
cubff, indipendentemente dal numero di programmi nella soup, dal numero di
epoche o dallo script che lancia la simulazione (reseed_and_track,
runit.py, cubff_example.py, un tuo script custom, ...).

Idea generale
-------------
`cubff.LanguageInterface.RunSimulation` chiama ad ogni `callback_interval`
epoche una funzione Python passandole un oggetto `SimulationState`. Questo
modulo fornisce:

1. `RunLogger`: un wrapper "drop-in" per quella callback. Si comporta come
   una normale callback (accetta `state`, ritorna un bool per fermare la
   simulazione) ma in più registra ad ogni chiamata un piccolo dizionario
   di metriche riassuntive, tutte espresse in forma "intensiva" (rapporti,
   frequenze relative, bit/byte) e quindi confrontabili a prescindere dalla
   scala (num_programs, num epoche, lunghezza dei programmi) dell'esperimento
   -- a differenza per esempio di `total_ops` o della lunghezza assoluta del
   soup, che dipendono direttamente da quei parametri.

2. `metrics_from_state`: la funzione pura che estrae le metriche da un
   singolo `SimulationState`, nel caso si preferisca scrivere la propria
   callback (es. dentro `reseed_and_track`) invece di usare `RunLogger`.

3. `load_records_from_jsonl` / `save_records_to_jsonl`: per chi logga su
   file (come fa già `runit.py` stampando una riga JSON per epoca) invece
   di tenere i record in memoria.

4. `plot_run`: produce una figura riassuntiva a piu' pannelli, ispirata
   alle Figure 1/5/6 del paper (Aguera y Arcas et al., "Computational
   Life"), usando solo le metriche gia' calcolate internamente da cubff
   (niente tracer tokens, che richiederebbero di rifare build del
   simulatore).

5. `plot_runs`: sovrappone piu' run (es. seed diversi, o mutation rate
   diversi) sullo stesso set di assi, per confronti come l'ablation sul
   rumore di fondo (Figura 6) o la distribuzione su tante run (Figura 5).

Metriche raccolte (tutte scale-free)
-------------------------------------
- higher_entropy : "high-order entropy" del paper (Shannon entropy - proxy
  di Kolmogorov complexity via compressione brotli). E' la metrica di
  complessita' principale usata per rilevare la transizione di stato.
- h0              : Shannon entropy (per-byte) del soup.
- brotli_bpb      : bit per byte del soup dopo compressione (proxy della
  complessita' di Kolmogorov normalizzata).
- top1_freq       : frequenza relativa (in [0,1]) del byte/token piu'
  comune nel soup -- analogo, in forma normalizzata, al crollo dei
  "unique tokens" di Figura 1: se un self-replicator prende il
  sopravvento, questa frequenza sale verso 1.
- top16_freq      : somma delle frequenze relative dei 16 byte piu' comuni
  (i "frequent_bytes" esposti da cubff) -- analogo del "top32 tokens" di
  Figura 1, ma normalizzato e quindi confrontabile fra run con soup di
  dimensioni diverse.
- frac_selfrep    : frazione (in [0,1]) di programmi nel soup il cui
  replication count supera la soglia `kSelfrepThreshold` -- versione
  normalizzata del conteggio assoluto di self-replicatori.
- mops_s          : velocita' di simulazione (solo informativo/performance,
  non e' una metrica del paper: utile per capire se una run e' bloccata,
  ma dipende dall'hardware, quindi non viene usato per confronti fra run).

Esempio d'uso rapido
---------------------
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bin"))
    import cubff
    from run_metrics import RunLogger, plot_run

    language = cubff.GetLanguage("bff_noheads")
    params = cubff.SimulationParams()
    params.num_programs = 131072
    params.seed = 0
    params.eval_selfrep = True  # necessario per popolare replication_per_prog

    logger = RunLogger(
        stop_fn=lambda state: state.epoch > 5000,
        selfrep_threshold=cubff.kSelfrepThreshold,
    )
    language.RunSimulation(params, None, logger)

    plot_run(logger.records, title="bff_noheads, seed=0")

Uso con reseed_and_track (o qualunque script che gia' logga JSON-lines)
-------------------------------------------------------------------------
    from run_metrics import load_records_from_jsonl, plot_run
    records = load_records_from_jsonl("path/to/log.jsonl")
    plot_run(records, title="run recuperata da log")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import matplotlib.pyplot as plt

from paper_constants import (
    PAPER_TRANSITION_ENTROPY_THRESHOLD,
    PAPER_SEEDED_TRANSITION_RATE,
    PAPER_SEEDED_EPOCHS,
)


# --------------------------------------------------------------------------
# Estrazione metriche da un SimulationState
# --------------------------------------------------------------------------


def metrics_from_state(state, selfrep_threshold: Optional[int] = None) -> dict:
    """Estrae un dizionario di metriche scale-free da un SimulationState.

    `selfrep_threshold` va passato (es. `cubff.kSelfrepThreshold`) se si
    vuole calcolare `frac_selfrep`; se None, o se `state.replication_per_prog`
    e' vuoto (perche' `params.eval_selfrep` era False), il campo sara' NaN.
    """
    frequent = list(getattr(state, "frequent_bytes", []) or [])
    top1_freq = frequent[0][1] if frequent else float("nan")
    top16_freq = sum(f for _, f in frequent) if frequent else float("nan")

    replication = list(getattr(state, "replication_per_prog", []) or [])
    if replication and selfrep_threshold is not None:
        frac_selfrep = sum(1 for c in replication if c >= selfrep_threshold) / len(
            replication
        )
    else:
        frac_selfrep = float("nan")

    return dict(
        epoch=state.epoch,
        higher_entropy=state.higher_entropy,
        h0=state.h0,
        brotli_bpb=state.brotli_bpb,
        top1_freq=top1_freq,
        top16_freq=top16_freq,
        frac_selfrep=frac_selfrep,
        mops_s=getattr(state, "mops_s", float("nan")),
    )


# --------------------------------------------------------------------------
# Callback riutilizzabile
# --------------------------------------------------------------------------


@dataclass
class RunLogger:
    """Callback "drop-in" per `LanguageInterface.RunSimulation`.

    Puo' incapsulare una callback/condizione di stop gia' esistente
    (`stop_fn`): RunLogger continuera' a inoltrarle lo stato e a
    propagarne il valore di ritorno, aggiungendo solo la registrazione
    delle metriche.
    """

    stop_fn: Optional[Callable[[object], bool]] = None
    selfrep_threshold: Optional[int] = None
    verbose: bool = False
    records: list = field(default_factory=list)

    def __call__(self, state) -> bool:
        record = metrics_from_state(state, self.selfrep_threshold)
        self.records.append(record)
        if self.verbose:
            print(json.dumps(record))
        if self.stop_fn is not None:
            return bool(self.stop_fn(state))
        return False


# --------------------------------------------------------------------------
# Salvataggio / caricamento log
# --------------------------------------------------------------------------


def save_records_to_jsonl(records: Sequence[dict], path: str) -> None:
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_records_from_jsonl(path: str) -> list:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------
# State transition detection (as in Figure 1 of the paper)
# --------------------------------------------------------------------------


def detect_transition_epoch(
    records: Sequence[dict],
    entropy_threshold: float = PAPER_TRANSITION_ENTROPY_THRESHOLD,
    x_key: str = "epoch",
) -> Optional[int]:
    """Primo valore di `x_key` in cui `higher_entropy` supera `entropy_threshold`.

    La soglia di default (1.0) e' quella usata dal paper stesso per questo
    identico scopo (vedi Figura 6: "distribution of the number of epochs
    needed for complexity to reach AT LEAST 1" - non 2, vedi
    paper_constants.PAPER_TRANSITION_ENTROPY_THRESHOLD per i dettagli).

    La soglia è sulla metrica di entropia (non su un conteggio assoluto),
    quindi il rilevamento resta valido a prescindere dalla dimensione della
    soup o dal numero di epoche della run. `x_key` permette di riusare
    questa funzione anche per run non basate su "epoca" (es. una
    simulazione a tape singolo, dove l'unita' naturale di progresso e' il
    numero di istruzioni eseguite): basta che i record abbiano una chiave
    con quel nome.
    """
    for r in records:
        h = r.get("higher_entropy")
        if h is not None and not math.isnan(h) and h > entropy_threshold:
            return r[x_key]
    return None


# --------------------------------------------------------------------------
# Plot di una singola run
# --------------------------------------------------------------------------


def plot_run(
    records: Sequence[dict],
    title: Optional[str] = None,
    entropy_threshold: float = PAPER_TRANSITION_ENTROPY_THRESHOLD,
    log_x: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    x_key: str = "epoch",
    x_label: str = "epoca",
):
    """Figura riassuntiva a 3 pannelli per una singola run, ispirata alle
    Figure 1/2 del paper ma costruita solo con metriche gia' normalizzate,
    quindi confrontabile fra run di scala diversa.

    Pannello 1 - Complessita': higher_entropy (metrica principale del
        paper) e h0 (Shannon entropy) nel tempo.
    Pannello 2 - Concentrazione del soup: frequenza del token piu' comune
        e dei 16 token piu' comuni -- versione normalizzata del crollo dei
        "unique tokens" osservato in Figura 1.
    Pannello 3 - Frazione di self-replicatori nel soup (se disponibile).

    Se viene rilevata una transizione di stato (vedi
    `detect_transition_epoch`), viene marcata con una linea verticale su
    tutti i pannelli.

    `x_key`/`x_label` permettono di riusare questa funzione per run la cui
    unita' naturale di progresso non e' "epoca" (es. il long-tape setup,
    dove si conta il numero di istruzioni eseguite): basta che ogni record
    abbia una chiave con nome `x_key`.
    """
    if not records:
        raise ValueError("No records to plot: the list is empty.")

    epochs = [r[x_key] for r in records]
    transition_epoch = detect_transition_epoch(records, entropy_threshold, x_key=x_key)

    has_selfrep = any(
        not math.isnan(r.get("frac_selfrep", float("nan"))) for r in records
    )
    n_panels = 3 if has_selfrep else 2

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9, 3.0 * n_panels), sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    # Panel 1: complexity
    ax = axes[0]
    ax.plot(epochs, [r["higher_entropy"] for r in records], label="high-order entropy", color="crimson")
    ax.plot(epochs, [r["h0"] for r in records], label="Shannon entropy (h0)", color="steelblue", alpha=0.7)
    ax.set_ylabel("bits / byte")
    ax.set_title(title or "Run evolution")
    ax.legend(loc="best", fontsize=8)

    # Panel 2: soup concentration (normalized proxy for "unique tokens")
    ax = axes[1]
    ax.plot(epochs, [r["top1_freq"] for r in records], label="most common token frequency", color="darkorange")
    ax.plot(epochs, [r["top16_freq"] for r in records], label="top-16 token frequency", color="olive", alpha=0.8)
    ax.set_ylabel("fraction of soup")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=8)

    # Panel 3 (optional): fraction of self-replicators
    if has_selfrep:
        ax = axes[2]
        ax.plot(epochs, [r["frac_selfrep"] for r in records], label="self-replicator fraction", color="seagreen")
        ax.set_ylabel("fraction of soup")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="best", fontsize=8)

    for ax in axes:
        if log_x:
            ax.set_xscale("log")
        if transition_epoch is not None:
            ax.axvline(transition_epoch, color="black", linestyle="--", linewidth=1, alpha=0.6)

    if transition_epoch is not None:
        axes[0].annotate(
            f"transition ~{x_label} {transition_epoch}",
            xy=(transition_epoch, axes[0].get_ylim()[1]),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=8,
        )

    axes[-1].set_xlabel(x_label)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Confronto fra piu' run (es. seed diversi, o mutation rate diversi)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Confronto con l'esperimento "seeded" del paper (par. 2.1, testo dopo Fig.5)
# --------------------------------------------------------------------------

# Il paper riporta: seedando la soup con UN autoreplicatore scritto a mano
# (quello di Figura 4) e facendo girare la simulazione per 128 epoche, si
# osserva una transizione di stato nel 22% delle run. E' il numero di
# riferimento naturale per "questo replicator si comporta come nel paper?".
# (valori definiti una sola volta in paper_constants.py, importati sopra
# insieme alla soglia di transizione - non ridefiniti qui per evitare che
# le due copie possano andare fuori sincrono)

# La logica per riprodurre questo esperimento (piazzare il replicator,
# far girare N run, contare le transizioni) vive ora in un unico posto:
# reseed_and_track.py (run_seeded_run + il loop in main(), via --runs).


def plot_runs(
    runs: Sequence[Sequence[dict]],
    labels: Optional[Sequence[str]] = None,
    metric: str = "higher_entropy",
    entropy_threshold: float = PAPER_TRANSITION_ENTROPY_THRESHOLD,
    log_x: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    x_key: str = "epoch",
    x_label: str = "epoca",
):
    """Sovrappone la stessa metrica per piu' run (es. per confrontare seed
    diversi o un ablation su un iperparametro, come nelle Figure 5/6 del
    paper). `metric` puo' essere una qualsiasi chiave prodotta da
    `metrics_from_state` (es. "higher_entropy", "top1_freq",
    "frac_selfrep", ...). `x_key`/`x_label` come in `plot_run`.
    """
    if not runs:
        raise ValueError("Nessuna run da plottare.")
    labels = labels or [f"run {i}" for i in range(len(runs))]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    transitions = []
    for records, label in zip(runs, labels):
        epochs = [r[x_key] for r in records]
        values = [r.get(metric, float("nan")) for r in records]
        ax.plot(epochs, values, label=label, alpha=0.85)
        t = detect_transition_epoch(records, entropy_threshold, x_key=x_key)
        if t is not None:
            transitions.append(t)

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric)
    ax.set_title(f"Run comparison: {metric}")
    if log_x:
        ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig
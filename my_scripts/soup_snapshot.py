#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soup_snapshot.py

Snapshot of a few programs of the primordial soup at selected epochs of one
run, used in Chapter 4 to illustrate what the soup looks like before, during
and after a state transition.

The run is simulated with the engine (bff_noheads, default mutation rate,
random pairing), so with the same seed it is exactly the run of Experiment 1.
The soup at epoch 0 is the engine's random initial soup, rebuilt by
token_soup.engine_initial_soup. At every requested epoch the script records:
  - the bytes of the programs to show (by default the first eight);
  - the self-replication score of each of them, computed by the engine;
  - the high-order entropy of the soup and the fractions of its bytes that
    are instructions and zeros.

Programs are drawn as in the debug output of the engine and in the figures of
Experiments 1 and 2: instructions as themselves, the zero byte as '0', every
other byte (a no-op) as a blank.

The engine measures the soup after epochs 1, 1 + k, 1 + 2k, ... (k = the
callback interval), so every requested epoch other than 0 must be of the form
1 + m*k.

Usage:
    python3 my_scripts/soup_snapshot.py --seed 171 \
        --epochs 0,2001,2497,4001 \
        --data soup_inspector_results/snapshot/seed171_snapshot.npz \
        --out images/soup_snapshot.pdf
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import token_soup as ts  # noqa: E402

COMMANDS = b"[]+-.,<>{}"
TAPE_SIZE = ts.TAPE_SIZE
INSTR_SHADE = (0.86, 0.90, 0.98)   # background of an instruction cell
ZERO_SHADE = (0.93, 0.93, 0.93)    # background of a zero cell


def cell_text(byte):
    if byte == 0:
        return "0"
    if byte in COMMANDS:
        return chr(byte)
    return " "


def row_text(program):
    return "".join(cell_text(b) for b in program)


def parse_programs(spec):
    """'0-7' or '0,5,9' -> list of program indices."""
    out = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def soup_fractions(soup):
    commands = np.frombuffer(COMMANDS, dtype=np.uint8)
    return float(np.isin(soup, commands).mean()), float((soup == 0).mean())


def simulate(args, programs):
    cubff = ts.import_cubff()
    language = cubff.GetLanguage(args.lang)
    epochs = sorted(set(args.epochs))
    for e in epochs:
        if e != 0 and (e - 1) % args.interval != 0:
            sys.exit(f"epoch {e} is not of the form 1 + m*{args.interval}")

    def score(program, epoch):
        return int(language.EvalParsedSelfrep(
            cubff.VectorUint8(bytes(program)), epoch, args.seed, False))

    records = []

    def record(epoch, soup, entropy):
        soup = soup.reshape(args.num_programs, TAPE_SIZE)
        chosen = soup[programs].copy()
        f_instr, f_zero = soup_fractions(soup)
        scores = [score(p, epoch) for p in chosen]
        records.append(dict(epoch=epoch, programs=chosen, scores=scores,
                            entropy=entropy, f_instr=f_instr, f_zero=f_zero))
        print(f"epoch {epoch:6d}: high-order entropy "
              f"{'n/a' if entropy is None else f'{entropy:.3f}'}, "
              f"instructions {100 * f_instr:.2f}% of bytes, "
              f"zeros {100 * f_zero:.2f}%", flush=True)
        for idx, p, s in zip(programs, chosen, scores):
            print(f"  {idx:6d} |{row_text(p)}| score {s:2d}", flush=True)

    if 0 in epochs:
        record(0, ts.engine_initial_soup(args.seed, args.num_programs), None)
    last = max(epochs)
    if last == 0:
        return records

    params = cubff.SimulationParams()
    params.num_programs = args.num_programs
    params.seed = args.seed
    params.callback_interval = args.interval
    params.eval_selfrep = False

    def callback(state):
        if state.epoch in epochs:
            record(state.epoch, np.array(state.soup, dtype=np.uint8),
                   float(state.higher_entropy))
        return state.epoch >= last

    language.RunSimulation(params, None, callback)
    return records


def save(records, programs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path,
             epochs=np.array([r["epoch"] for r in records]),
             program_index=np.array(programs),
             programs=np.stack([r["programs"] for r in records]),
             scores=np.array([r["scores"] for r in records]),
             entropy=np.array([np.nan if r["entropy"] is None
                               else r["entropy"] for r in records]),
             f_instr=np.array([r["f_instr"] for r in records]),
             f_zero=np.array([r["f_zero"] for r in records]))
    print(f"written {path}")


def load(path):
    d = np.load(path)
    records = []
    for i, e in enumerate(d["epochs"]):
        ent = float(d["entropy"][i])
        records.append(dict(epoch=int(e), programs=d["programs"][i],
                            scores=list(d["scores"][i]),
                            entropy=None if np.isnan(ent) else ent,
                            f_instr=float(d["f_instr"][i]),
                            f_zero=float(d["f_zero"][i])))
    return records, list(d["program_index"])


def draw(records, programs, threshold, out):
    n_prog = len(programs)
    rows_per_panel = n_prog + 1.6            # programs + title line
    n_rows = rows_per_panel * len(records)
    label_w, score_w = 5.0, 6.0
    cell_w, cell_h = 1.0, 1.9
    width_in = 6.3
    unit = width_in / (label_w + TAPE_SIZE * cell_w + score_w)
    fig, ax = plt.subplots(figsize=(width_in, (n_rows + 0.5) * cell_h * unit))
    ax.set_xlim(-label_w, TAPE_SIZE * cell_w + score_w)
    ax.set_ylim(-(n_rows + 0.2) * cell_h, 0.3 * cell_h)
    ax.axis("off")
    mono = {"family": "DejaVu Sans Mono", "size": 6.2}

    y = 0.0
    for r in records:
        parts = [f"epoch {r['epoch']:,}"]
        if r["entropy"] is not None:
            parts.append(f"high-order entropy {r['entropy']:.2f}")
        parts.append(f"instructions {100 * r['f_instr']:.1f}% of the bytes")
        parts.append(f"zeros {100 * r['f_zero']:.1f}%")
        ax.text(0, y - 0.6 * cell_h, ",  ".join(parts), ha="left",
                va="center", fontsize=6.6, fontweight="bold")
        ax.text(TAPE_SIZE * cell_w + score_w / 2, y - 0.6 * cell_h, "score",
                ha="center", va="center", fontsize=6.0, color="0.3")
        y -= 1.4 * cell_h
        for idx, prog, sc in zip(programs, r["programs"], r["scores"]):
            for x, b in enumerate(prog):
                shade = (INSTR_SHADE if b in COMMANDS
                         else ZERO_SHADE if b == 0 else None)
                if shade is not None:
                    ax.add_patch(Rectangle((x * cell_w, y - cell_h), cell_w,
                                           cell_h, facecolor=shade,
                                           edgecolor="none"))
                t = cell_text(b)
                if t != " ":
                    ax.text((x + 0.5) * cell_w, y - cell_h / 2, t,
                            ha="center", va="center",
                            color="0.45" if b == 0 else "black", **mono)
            ax.add_patch(Rectangle((0, y - cell_h), TAPE_SIZE * cell_w,
                                   cell_h, facecolor="none",
                                   edgecolor="0.75", linewidth=0.3))
            ax.text(-0.8, y - cell_h / 2, str(idx), ha="right", va="center",
                    fontsize=5.6, color="0.4")
            ax.text(TAPE_SIZE * cell_w + score_w / 2, y - cell_h / 2,
                    f"{sc}", ha="center", va="center", fontsize=6.0,
                    fontweight="bold" if sc >= threshold else "normal",
                    color="black" if sc >= threshold else "0.45")
            y -= cell_h
        y -= 0.2 * cell_h
    fig.savefig(out, bbox_inches="tight")
    print(f"written {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--seed", type=int, default=171)
    ap.add_argument("--num-programs", type=int, default=1 << 17)
    ap.add_argument("--lang", default="bff_noheads")
    ap.add_argument("--epochs", default="0,2001,2497,4001",
                    help="comma-separated epochs (0 or 1 + m*interval)")
    ap.add_argument("--interval", type=int, default=16,
                    help="callback interval of the engine")
    ap.add_argument("--programs", default="0-7",
                    help="indices of the programs to show, e.g. 0-7")
    ap.add_argument("--data", default="soup_inspector_results/snapshot/"
                    "seed171_snapshot.npz")
    ap.add_argument("--out", default="images/soup_snapshot.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from --data without simulating")
    args = ap.parse_args()
    args.epochs = [int(e) for e in args.epochs.split(",")]
    threshold = 5   # kSelfrepThreshold of the engine

    if args.plot_only:
        records, programs = load(args.data)
    else:
        programs = parse_programs(args.programs)
        records = simulate(args, programs)
        save(records, programs, args.data)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    draw(records, programs, threshold, args.out)


if __name__ == "__main__":
    main()

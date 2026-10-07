#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transition_survey.py

Reproduces the "fraction of runs with a state transition within N epochs"
measurement of the paper (Section 2.1, Fig. 5 and the 0.024% row of Fig. 6):
for each seed, a fresh random soup is simulated and the first epoch at which
the soup's higher-order entropy reaches the transition threshold (1.0) is
recorded.

Differences from soup_inspector.py:
  * eval_selfrep is NEVER enabled (no full-soup self-replication checks, no
    mid-run toggling), only the aggregate entropy the engine already
    computes is read. Cheaper, and immune to the toggling issues.
  * No checkpoints, no replicator files: one CSV row per seed.
  * Resumable: seeds already present in the output CSV are skipped, so the
    survey can be interrupted and restarted, or split across several
    processes by seed range (one --out file per process).

Defaults follow cubff's bin/main (the tool the paper's experiments were run
with): entropy measured every 64 epochs (bin/main --print_interval default)
and mutation_prob = 1/4096 = 2^-12 (the paper's "0.024%", rounded).
The callback interval only controls how often entropy is MEASURED: in
common_language.h the simulation's randomness depends only on (seed, epoch),
so the trajectory is identical for any callback interval.

Typical usage (from the cubff/cubff folder, after building with PYTHON=1):

    # pilot run: one seed, measure wall time
    python3 -u transition_survey.py --seeds 1 --out pilot.csv

    # full survey, split in 4 processes
    python3 -u transition_survey.py --seeds 1-250    --out part1.csv &
    python3 -u transition_survey.py --seeds 251-500  --out part2.csv &
    python3 -u transition_survey.py --seeds 501-750  --out part3.csv &
    python3 -u transition_survey.py --seeds 751-1000 --out part4.csv &

    # summary over all parts
    python3 transition_survey.py --summarize part1.csv part2.csv part3.csv part4.csv

    # Fig. 7 "seeded": random soup plus the paper's hand-written
    # self-replicator (the one of its Fig. 4) placed in a random position,
    # run for 128 epochs
    python3 -u transition_survey.py --seeds 1-1000 --max-epochs 128 \
        --full-run --initial-program paper_fig4 --out survey_fig7_seeded.csv
"""

import argparse
import csv
import math
import os
import struct
import sys
import tempfile
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from paper_constants import (
    PAPER_NUM_PROGRAMS,
    PAPER_TRANSITION_ENTROPY_THRESHOLD,
)

CSV_FIELDS = [
    "seed", "lang", "num_programs", "mutation_prob", "permute_programs",
    "fixed_shuffle",
    "max_epochs", "callback_interval", "entropy_threshold",
    "transition_epoch", "entropy_at_transition",
    "last_epoch", "final_entropy", "wall_seconds",
    # Appended last so that the column positions of older CSVs do not change.
    "initial_program", "initial_program_index",
]

# Fig. 6, 0.024% row: fraction of runs whose FIRST crossing of complexity >= 1
# falls in each 2000-epoch bucket; the last value is "no transition in 16k".
PAPER_FIG6_DEFAULT_ROW = [0.09, 0.08, 0.04, 0.05, 0.04, 0.03, 0.04, 0.03, 0.60]


TAPE_SIZE = 64  # bytes per program (kSingleTapeSize in cubff)

# Hand-written self-replicator shown in Fig. 4 of the paper and used for the
# "seeded" condition of its Fig. 7: a 64-byte palindrome made of the copy
# loop, 48 non-coding bytes (spaces in the paper's figure) and the mirrored
# loop.
NAMED_PROGRAMS = {
    "paper_fig4": b"[[{.>]-]" + b" " * 48 + b"]-]>.{[[",
}


def resolve_initial_program(spec):
    """'paper_fig4' or 128 hex digits -> the 64 bytes of the program."""
    if spec in NAMED_PROGRAMS:
        program = NAMED_PROGRAMS[spec]
    else:
        try:
            program = bytes.fromhex(spec)
        except ValueError:
            sys.exit(f"--initial-program: '{spec}' is neither a known name "
                     f"({', '.join(NAMED_PROGRAMS)}) nor a hex string")
    if len(program) != TAPE_SIZE:
        sys.exit(f"--initial-program must be {TAPE_SIZE} bytes, "
                 f"got {len(program)}")
    return program


MASK64 = (1 << 64) - 1


def splitmix64(x):
    """Same generator as SplitMix64() in cubff's common_language.h."""
    z = (x + 0x9e3779b97f4a7c15) & MASK64
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK64
    return z ^ (z >> 31)


def engine_initial_soup(seed, num_programs):
    """The random soup the engine itself creates for this seed, byte for byte
    (InitPrograms in common_language.h), as a numpy uint8 array."""
    import numpy as np
    init_seed = splitmix64(splitmix64(seed) ^ splitmix64(0))
    base = (TAPE_SIZE * num_programs * init_seed) & MASK64
    with np.errstate(over="ignore"):  # 64-bit wrap-around is intended
        z = np.arange(TAPE_SIZE * num_programs, dtype=np.uint64)
        z = z + np.uint64(base)
        z = z + np.uint64(0x9e3779b97f4a7c15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(0xFF)).astype(np.uint8)


def seeded_index(seed, num_programs):
    """Position of the added program: uniform over the soup, and a fixed
    function of the run's seed so that every run can be reproduced."""
    return splitmix64(splitmix64(seed) ^ splitmix64(0x5EEDED)) % num_programs


def write_initial_checkpoint(soup, num_programs, tmp_dir):
    """Write `soup` as a cubff checkpoint at epoch 0 and return its path.
    Layout (see RunSimulation): reset_index, num_programs, epoch as 64-bit
    integers, followed by the soup bytes."""
    fd, path = tempfile.mkstemp(prefix="seeded_", suffix=".dat", dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<QQQ", 1, num_programs, 0))
        f.write(soup.tobytes())
    return path


def check_initial_soup(cubff, language, args, seed):
    """Make sure that loading our own copy of the initial soup is equivalent
    to letting the engine initialize it: after one epoch the two soups must
    be identical. Guards against any mismatch between engine_initial_soup()
    and the engine actually in use."""
    def soup_after_first_epoch(load_from):
        params = cubff.SimulationParams()
        params.num_programs = args.num_programs
        params.seed = seed
        params.mutation_prob = args.mutation_raw
        params.callback_interval = args.callback_interval
        params.permute_programs = not args.no_permute
        params.fixed_shuffle = args.fixed_shuffle
        params.eval_selfrep = False
        if load_from:
            params.load_from = load_from
        out = {}

        def callback(state):
            out["soup"] = bytes(state.soup)
            return True
        language.RunSimulation(params, None, callback)
        return out["soup"]

    path = write_initial_checkpoint(
        engine_initial_soup(seed, args.num_programs), args.num_programs,
        args.tmp_dir)
    try:
        same = soup_after_first_epoch(None) == soup_after_first_epoch(path)
    finally:
        os.remove(path)
    if not same:
        sys.exit("The initial soup built by this script does not match the "
                 "engine's own initialization. Nothing was run.")
    print("initial soup check passed: loading our copy of the random soup "
          "is equivalent to the engine's own initialization")


def import_cubff():
    try:
        from bin import cubff
        return cubff
    except ImportError as exc:
        sys.stderr.write(
            "Could not import bin/cubff*.so. Build with PYTHON=1 "
            "(e.g. 'make CUDA=0 PYTHON=1') and run from cubff/cubff.\n"
            f"Original error: {exc}\n")
        sys.exit(1)


def encode_mutation_prob(p):
    """Same encoding used by soup_inspector.py: probability * 2^30."""
    return round(p * (1 << 30))


def parse_seeds(spec):
    """'1-1000', '5', '1,3,10-20' -> sorted list of ints."""
    seeds = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    return sorted(seeds)


def config_key(row):
    """Fields that must match for two rows to belong to the same experiment."""
    return (row["lang"], int(row["num_programs"]), float(row["mutation_prob"]),
            str(row["permute_programs"]), str(row.get("fixed_shuffle", False)),
            int(row["max_epochs"]), float(row["entropy_threshold"]),
            row.get("initial_program") or "")


def read_rows(paths):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def check_mutation_encoding(cubff, raw):
    """The paper uses the code's default mutation rate (1<<18 out of 1<<30,
    i.e. 2^-12). Report whether the rate about to be used matches it."""
    engine_default = cubff.SimulationParams().mutation_prob
    if raw != engine_default:
        print(f"NOTE: mutation_prob={raw}/2^30 differs from the engine "
              f"default {engine_default}/2^30 (paper default). Fine for "
              f"ablations, not for reproducing the default setting.")
    else:
        print(f"mutation_prob = {raw}/2^30 = engine/paper default")


def run_one(cubff, language, seed, args, trace_path=None):
    params = cubff.SimulationParams()
    params.num_programs = args.num_programs
    params.seed = seed
    params.mutation_prob = args.mutation_raw
    params.callback_interval = args.callback_interval
    params.permute_programs = not args.no_permute
    params.fixed_shuffle = args.fixed_shuffle
    params.eval_selfrep = False  # never touched again

    result = {"transition_epoch": None, "entropy_at_transition": None,
              "last_epoch": 0, "final_entropy": None,
              "initial_program_index": None}
    checkpoint = None
    if args.initial_program_bytes is not None:
        # "Seeded" run: the engine's own random soup for this seed, with one
        # program replaced by the given one. The soup is handed to the engine
        # as a checkpoint at epoch 0, so the rest of the run (pairing and
        # mutations) is exactly that of the unseeded run with the same seed.
        import numpy as np
        index = (args.initial_program_index
                 if args.initial_program_index is not None
                 else seeded_index(seed, args.num_programs))
        soup = engine_initial_soup(seed, args.num_programs)
        soup[index * TAPE_SIZE:(index + 1) * TAPE_SIZE] = np.frombuffer(
            args.initial_program_bytes, dtype=np.uint8)
        checkpoint = write_initial_checkpoint(soup, args.num_programs,
                                              args.tmp_dir)
        params.load_from = checkpoint
        result["initial_program_index"] = index
    trace = open(trace_path, "w") if trace_path else None
    if trace:
        trace.write("epoch,higher_entropy\n")

    def callback(state):
        epoch = state.epoch
        h = state.higher_entropy
        result["last_epoch"] = epoch
        result["final_entropy"] = h
        if trace:
            trace.write(f"{epoch},{h:.6f}\n")
        if (result["transition_epoch"] is None
                and h >= args.entropy_threshold
                and epoch <= args.max_epochs):
            result["transition_epoch"] = epoch
            result["entropy_at_transition"] = h
            if not args.full_run:
                return True  # stop: this seed is done
        return epoch >= args.max_epochs

    t0 = time.time()
    try:
        language.RunSimulation(params, None, callback)
    finally:
        if trace:
            trace.close()
        if checkpoint:
            os.remove(checkpoint)
    result["wall_seconds"] = round(time.time() - t0, 1)
    return result


def run_survey(args):
    cubff = import_cubff()
    args.mutation_raw = encode_mutation_prob(args.mutation_prob)
    args.mutation_prob = args.mutation_raw / (1 << 30)  # effective value
    check_mutation_encoding(cubff, args.mutation_raw)
    args.initial_program_bytes = None
    if args.initial_program:
        args.initial_program_bytes = resolve_initial_program(
            args.initial_program)
        if (args.initial_program_index is not None and not
                0 <= args.initial_program_index < args.num_programs):
            sys.exit("--initial-program-index out of range")
    language = cubff.GetLanguage(args.lang)
    if hasattr(cubff, "ResetColors"):
        cubff.ResetColors()

    done = {int(r["seed"]) for r in read_rows([args.out])}
    todo = [s for s in parse_seeds(args.seeds) if s not in done]
    print(f"lang={args.lang} programs={args.num_programs} "
          f"mutation={args.mutation_prob} max_epochs={args.max_epochs} "
          f"callback_interval={args.callback_interval} "
          f"initial_program={args.initial_program or 'none'} | "
          f"{len(done)} seeds already in '{args.out}', {len(todo)} to run")

    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)
    if args.initial_program and todo:
        check_initial_soup(cubff, language, args, todo[0])

    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for i, seed in enumerate(todo, 1):
            trace_path = (os.path.join(args.trace_dir, f"seed{seed}.csv")
                          if args.trace_dir else None)
            r = run_one(cubff, language, seed, args, trace_path)
            writer.writerow({
                "seed": seed, "lang": args.lang,
                "num_programs": args.num_programs,
                "mutation_prob": args.mutation_prob,
                "permute_programs": not args.no_permute,
                "fixed_shuffle": args.fixed_shuffle,
                "max_epochs": args.max_epochs,
                "callback_interval": args.callback_interval,
                "entropy_threshold": args.entropy_threshold,
                "transition_epoch": ("" if r["transition_epoch"] is None
                                     else r["transition_epoch"]),
                "entropy_at_transition": ("" if r["entropy_at_transition"]
                                          is None else
                                          f"{r['entropy_at_transition']:.6f}"),
                "last_epoch": r["last_epoch"],
                "final_entropy": ("" if r["final_entropy"] is None
                                  else f"{r['final_entropy']:.6f}"),
                "wall_seconds": r["wall_seconds"],
                "initial_program": args.initial_program or "",
                "initial_program_index": (
                    "" if r["initial_program_index"] is None
                    else r["initial_program_index"]),
            })
            f.flush()  # keep the CSV consistent if the process is killed
            status = (f"threshold first reached at epoch "
                      f"{r['transition_epoch']}"
                      if r["transition_epoch"] is not None
                      else f"threshold not reached by epoch {args.max_epochs}")
            final = ("n/a" if r["final_entropy"] is None
                     else f"{r['final_entropy']:.3f}")
            print(f"[{i}/{len(todo)}] seed {seed}: {status}, entropy "
                  f"{final} at epoch {r['last_epoch']} "
                  f"({r['wall_seconds']} s)")


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(paths, bucket=2000):
    rows = read_rows(paths)
    if not rows:
        print("No rows found.")
        return
    groups = {}
    for r in rows:
        groups.setdefault(config_key(r), {})[int(r["seed"])] = r  # dedup seeds

    for key, by_seed in groups.items():
        lang, n_prog, mut, permute, fixed, max_epochs, thr, initial = key
        n = len(by_seed)
        epochs = [int(r["transition_epoch"]) for r in by_seed.values()
                  if r["transition_epoch"] != ""]
        k = len(epochs)
        lo, hi = wilson_interval(k, n)
        print(f"\n=== lang={lang} programs={n_prog} mutation={mut} "
              f"permute={permute} fixed_shuffle={fixed} "
              f"max_epochs={max_epochs} threshold={thr} "
              f"initial_program={initial or 'none'} ===")
        print(f"runs: {n} | with transition: {k} ({k / n:.1%}, "
              f"95% CI {lo:.1%}-{hi:.1%})")
        # End-of-run criterion of the paper's Fig. 7. Only meaningful for
        # surveys made with --full-run: otherwise a transitioning seed stops
        # at the transition and its last entropy is not the end-of-run one.
        full = [r for r in by_seed.values()
                if r["final_entropy"] != ""
                and int(r["last_epoch"]) >= max_epochs]
        if len(full) == n:
            kf = sum(float(r["final_entropy"]) >= thr for r in full)
            lof, hif = wilson_interval(kf, n)
            print(f"above threshold at the end of the run: {kf} "
                  f"({kf / n:.1%}, 95% CI {lof:.1%}-{hif:.1%})")

        n_buckets = math.ceil(max_epochs / bucket)
        counts = [0] * n_buckets
        for e in epochs:
            counts[min(max(e - 1, 0) // bucket, n_buckets - 1)] += 1
        labels = [f"<={(b + 1) * bucket}" for b in range(n_buckets)] + ["N/A"]
        fractions = [c / n for c in counts] + [(n - k) / n]
        print("bucket   " + " ".join(f"{l:>7}" for l in labels))
        print("ours     " + " ".join(f"{x:7.2f}" for x in fractions))
        if (abs(mut - 2 ** -12) < 1e-12 and permute == "True"
                and fixed == "False" and max_epochs == 16000
                and bucket == 2000 and not initial):
            print("paper    " + " ".join(f"{x:7.2f}"
                                         for x in PAPER_FIG6_DEFAULT_ROW))


def main():
    parser = argparse.ArgumentParser(
        description="Entropy-only survey of state transitions over many "
                    "seeds (paper Section 2.1, Fig. 5/6).")
    parser.add_argument("--summarize", nargs="+", metavar="CSV",
                        help="Only print a summary of the given CSV files.")
    parser.add_argument("--seeds", default="1-1000",
                        help="Seeds to run, e.g. '1-1000' or '1,5,10-20'.")
    parser.add_argument("--out", default="transition_survey.csv",
                        help="Output CSV (appended to; existing seeds are "
                             "skipped).")
    parser.add_argument("--lang", default="bff_noheads")
    parser.add_argument("--num-programs", type=int,
                        default=PAPER_NUM_PROGRAMS)
    parser.add_argument("--max-epochs", type=int, default=16000)
    parser.add_argument("--mutation-prob", type=float,
                        default=2 ** -12,
                        help="Per-byte mutation probability as a fraction. "
                             "Default 2^-12 = 1/4096 (~0.0244%%), the cubff "
                             "default the paper reports as 0.024%%.")
    parser.add_argument("--no-permute", action="store_true",
                        help="Cyclic neighbour pairing (bin/main "
                             "--nopermute_programs). NOT the paper's "
                             "long-no-noise setting: use --fixed-shuffle.")
    parser.add_argument("--fixed-shuffle", action="store_true",
                        help="Deterministic shuffle pattern (bin/main "
                             "--fixed_shuffle), as in the paper's "
                             "long-no-noise runs together with "
                             "--mutation-prob 0. Needs a power-of-two "
                             "--num-programs.")
    parser.add_argument("--entropy-threshold", type=float,
                        default=PAPER_TRANSITION_ENTROPY_THRESHOLD)
    parser.add_argument("--callback-interval", type=int, default=64,
                        help="Epochs between entropy evaluations (default 64, "
                             "as bin/main --print_interval). Does not change "
                             "the dynamics, only the time resolution of the "
                             "recorded transition epoch.")
    parser.add_argument("--full-run", action="store_true",
                        help="Do not stop at the transition: always run "
                             "--max-epochs (needed for Fig. 5 curves).")
    parser.add_argument("--initial-program", default=None,
                        help="Replace one program of the random initial "
                             "soup with the given one (the paper's 'seeded' "
                             "runs). Either a known name ('paper_fig4', the "
                             "hand-written self-replicator of the paper's "
                             "Fig. 4) or the 64 bytes as 128 hex digits.")
    parser.add_argument("--initial-program-index", type=int, default=None,
                        help="Position (program index) of --initial-program "
                             "in the soup. Default: a random position, "
                             "different for every seed, as in the paper.")
    parser.add_argument("--tmp-dir", default=None,
                        help="Directory for the temporary initial-soup files "
                             "of seeded runs (default: system temp dir).")
    parser.add_argument("--trace-dir", default=None,
                        help="If set, write the entropy trace of every seed "
                             "to <trace-dir>/seed<N>.csv.")
    parser.add_argument("--bucket", type=int, default=2000,
                        help="Bucket width for --summarize (default 2000, "
                             "as in Fig. 6).")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.summarize, args.bucket)
        return

    if args.num_programs % 2:
        args.num_programs += 1
        sys.stderr.write(f"num-programs rounded up to {args.num_programs}\n")
    if args.fixed_shuffle and (args.num_programs & (args.num_programs - 1)):
        sys.exit("--fixed-shuffle needs a power-of-two --num-programs")

    run_survey(args)
    summarize([args.out], args.bucket)


if __name__ == "__main__":
    main()
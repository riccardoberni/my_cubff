#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial_survey.py

Runs of the spatial (2D) BFF soup of the paper (Section 2.2, Fig. 8): the
programs are arranged on a W x H grid and two programs can interact only if
their distance along each coordinate is at most --radius (2 in the paper).
With --height 1 the same script gives a 1D soup, and with --well-mixed the
grid is ignored and the programs are paired as in the standard primordial
soup (the width and height then only set the number of programs and the
shape of the snapshots), which is the zero-dimensional reference.

The grid is passed to the engine as a list of allowed interactions, built in
the same order as cubff's make_2d_pattern.py, so a run of this script is the
same run as `bin/main --interaction_pattern <file written by
make_2d_pattern.py>` with the same seed.

For every seed the script records, every --callback-interval epochs:
  * the high-order entropy of the whole grid (as transition_survey.py);
  * the number of programs that the engine's self-replication check
    (eval_selfrep, score >= kSelfrepThreshold = 5) marks as self-replicators,
    which measures how much of the grid has been taken over;
  * the number of pairs executed in the last epoch and, with
    --long-range-prob, how many of them were long-range.
and, every --snapshot-interval epochs, the whole grid (bytes and
self-replication scores) as a compressed .npz file, used for the images
(plot_spatial_snapshots.py) and for the spatial analyses.

Note on the self-replication check: the engine evaluates the programs in
blocks of 32 and skips the last num_programs % 32 of them (16 programs out of
32,400 on the 240 x 135 grid), whose scores are left undefined. The script
therefore uses only the first (num_programs // 32) * 32 programs for the
self-replicator counts and fractions, and stores a score of 0 for the others
in the snapshots.

--long-range-prob > 0 needs an engine built with long_range_prob.patch.

Outputs, in --out-dir:
  summary.csv              one row per seed (appended; seeds already present
                           are skipped, so the survey can be resumed)
  traces/seed<N>.csv       time series of the measures above
  snapshots/seed<N>_epoch<E>.npz
                           soup (H, W, 64) uint8, selfrep (H, W) uint8

Typical usage (from the repository root, on a GPU node):

    python3 -u my_scripts/spatial_survey.py --seeds 1-10 \
        --out-dir spatial_results/grid_240x135

    python3 my_scripts/spatial_survey.py --summarize \
        spatial_results/grid_240x135/summary.csv
"""

import argparse
import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np

from paper_constants import PAPER_TRANSITION_ENTROPY_THRESHOLD
from transition_survey import (
    TAPE_SIZE,
    encode_mutation_prob,
    engine_initial_soup,
    import_cubff,
    parse_seeds,
    resolve_initial_program,
    seeded_index,
    wilson_interval,
    write_initial_checkpoint,
)

# Fractions of the grid occupied by self-replicators whose first reaching
# epoch is stored in summary.csv (takeover milestones).
SELFREP_MILESTONES = [0.01, 0.1, 0.5, 0.9]

CONFIG_FIELDS = [
    "lang", "width", "height", "radius", "well_mixed", "mutation_prob",
    "long_range_prob",
    "max_epochs", "callback_interval", "entropy_threshold", "eval_selfrep",
    "initial_program", "initial_position",
]
SUMMARY_FIELDS = (
    ["seed"] + CONFIG_FIELDS +
    ["initial_program_index", "extinct", "transition_epoch",
     "entropy_at_transition",
     "last_epoch", "final_entropy", "final_selfrep_fraction"] +
    [f"epoch_selfrep_{int(f * 100)}pct" for f in SELFREP_MILESTONES] +
    ["wall_seconds"])
TRACE_FIELDS = ["epoch", "higher_entropy", "h0", "brotli_bpb", "num_selfrep",
                "selfrep_fraction", "num_pairs", "num_long_range_pairs"]


def grid_neighbours(width, height, radius):
    """Allowed partners of every program, in the same order as cubff's
    make_2d_pattern.py (program index = row * width + column). The order
    matters: the engine picks the neighbour by its position in the list."""
    neighbours = []
    for i in range(height):
        for j in range(width):
            row = []
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ii, jj = i + di, j + dj
                    if (0 <= ii < height and 0 <= jj < width
                            and (di != 0 or dj != 0)):
                        row.append(ii * width + jj)
            neighbours.append(row)
    return neighbours


def config_of(args):
    return {
        "lang": args.lang,
        "width": args.width,
        "height": args.height,
        "radius": args.radius,
        "well_mixed": args.well_mixed,
        "mutation_prob": args.mutation_prob,
        "long_range_prob": args.long_range_prob,
        "max_epochs": args.max_epochs,
        "callback_interval": args.callback_interval,
        "entropy_threshold": args.entropy_threshold,
        "eval_selfrep": not args.no_selfrep,
        "initial_program": args.initial_program or "",
        "initial_position": args.initial_position if args.initial_program else "",
    }


def same_config(row, config):
    for key, value in config.items():
        if isinstance(value, bool):
            if row[key] != str(value):
                return False
        elif isinstance(value, float):
            if abs(float(row[key]) - value) > 1e-15:
                return False
        elif isinstance(value, int):
            if int(row[key]) != value:
                return False
        elif row[key] != value:
            return False
    return True


def initial_index(args, seed, num_programs):
    if args.initial_program_index is not None:
        return args.initial_program_index
    if args.initial_position == "center":
        return (args.height // 2) * args.width + args.width // 2
    return seeded_index(seed, num_programs)


def run_one(cubff, language, neighbours, seed, args):
    num_programs = args.width * args.height
    params = cubff.SimulationParams()
    params.num_programs = num_programs
    params.seed = seed
    params.mutation_prob = args.mutation_raw
    params.callback_interval = args.callback_interval
    params.eval_selfrep = not args.no_selfrep
    if neighbours is not None:
        params.allowed_interactions = [cubff.VectorUint32(x)
                                       for x in neighbours]
    if args.long_range_prob > 0:
        params.long_range_prob = args.long_range_raw

    result = {"initial_program_index": "", "extinct": False,
              "transition_epoch": "",
              "entropy_at_transition": "", "last_epoch": 0,
              "final_entropy": "", "final_selfrep_fraction": ""}
    milestones = {f: "" for f in SELFREP_MILESTONES}

    checkpoint = None
    if args.initial_program_bytes is not None:
        # Seeded run: the engine's own random soup for this seed with one
        # program replaced, handed to the engine as a checkpoint at epoch 0
        # (same procedure as transition_survey.py).
        index = initial_index(args, seed, num_programs)
        soup = engine_initial_soup(seed, num_programs)
        soup[index * TAPE_SIZE:(index + 1) * TAPE_SIZE] = np.frombuffer(
            args.initial_program_bytes, dtype=np.uint8)
        checkpoint = write_initial_checkpoint(soup, num_programs, args.tmp_dir)
        params.load_from = checkpoint
        result["initial_program_index"] = index

    # Programs actually evaluated by the engine's self-replication check.
    num_checked = (num_programs // 32) * 32

    trace_path = os.path.join(args.out_dir, "traces", f"seed{seed}.csv")
    trace = open(trace_path, "w", newline="")
    writer = csv.writer(trace)
    writer.writerow(TRACE_FIELDS)

    def callback(state):
        epoch = state.epoch
        h = state.higher_entropy
        if params.eval_selfrep:
            scores = np.zeros(num_programs, dtype=np.int64)
            scores[:num_checked] = np.asarray(
                state.replication_per_prog)[:num_checked]
            num_selfrep = int((scores >= cubff.kSelfrepThreshold).sum())
            fraction = num_selfrep / num_checked
        else:
            scores, num_selfrep, fraction = None, "", ""
        num_pairs = getattr(state, "num_pairs", "")
        num_long = getattr(state, "num_long_range_pairs", "")
        writer.writerow([epoch, f"{h:.6f}", f"{state.h0:.6f}",
                         f"{state.brotli_bpb:.6f}", num_selfrep,
                         f"{fraction:.6f}" if fraction != "" else "",
                         num_pairs, num_long])
        trace.flush()

        result["last_epoch"] = epoch
        result["final_entropy"] = f"{h:.6f}"
        result["final_selfrep_fraction"] = (f"{fraction:.6f}"
                                            if fraction != "" else "")
        if result["transition_epoch"] == "" and h >= args.entropy_threshold:
            result["transition_epoch"] = epoch
            result["entropy_at_transition"] = f"{h:.6f}"
        if fraction != "":
            for f in SELFREP_MILESTONES:
                if milestones[f] == "" and fraction >= f:
                    milestones[f] = epoch

        if args.snapshot_interval and (epoch - 1) % args.snapshot_interval == 0:
            soup = np.frombuffer(bytes(state.soup), dtype=np.uint8)
            selfrep = (np.minimum(scores, 255).astype(np.uint8)
                       if scores is not None
                       else np.zeros(num_programs, dtype=np.uint8))
            np.savez_compressed(
                os.path.join(args.out_dir, "snapshots",
                             f"seed{seed}_epoch{epoch}.npz"),
                soup=soup.reshape(args.height, args.width, TAPE_SIZE),
                selfrep=selfrep.reshape(args.height, args.width),
                epoch=epoch, seed=seed)

        if (args.stop_fraction is not None and fraction != ""
                and fraction >= args.stop_fraction):
            return True
        if (args.stop_if_extinct_after is not None and fraction != ""
                and epoch >= args.stop_if_extinct_after and num_selfrep == 0):
            result["extinct"] = True
            return True
        return epoch >= args.max_epochs

    t0 = time.time()
    try:
        language.RunSimulation(params, None, callback)
    finally:
        trace.close()
        if checkpoint:
            os.remove(checkpoint)
    result["wall_seconds"] = round(time.time() - t0, 1)
    for f in SELFREP_MILESTONES:
        result[f"epoch_selfrep_{int(f * 100)}pct"] = milestones[f]
    return result


def run_survey(args):
    cubff = import_cubff()
    if args.long_range_prob > 0 and not hasattr(cubff.SimulationParams(),
                                                "long_range_prob"):
        sys.exit("--long-range-prob needs an engine built with "
                 "long_range_prob.patch")
    language = cubff.GetLanguage(args.lang)
    args.mutation_raw = encode_mutation_prob(args.mutation_prob)
    args.mutation_prob = args.mutation_raw / (1 << 30)  # effective value
    args.long_range_raw = round(args.long_range_prob * (1 << 30))
    args.long_range_prob = args.long_range_raw / (1 << 30)
    args.initial_program_bytes = (resolve_initial_program(args.initial_program)
                                  if args.initial_program else None)

    for sub in ("traces", "snapshots"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    config = config_of(args)

    done = set()
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                if not same_config(row, config):
                    sys.exit(f"{summary_path} contains runs with a different "
                             f"configuration (seed {row['seed']}). Use "
                             f"another --out-dir.")
                done.add(int(row["seed"]))

    if args.well_mixed:
        neighbours = None
        print(f"well-mixed soup of {args.width * args.height} programs, "
              f"mutation_prob {args.mutation_prob}")
    else:
        neighbours = grid_neighbours(args.width, args.height, args.radius)
        print(f"grid {args.width} x {args.height} = "
              f"{args.width * args.height} programs, radius {args.radius}, "
              f"long_range_prob {args.long_range_prob}, mutation_prob "
              f"{args.mutation_prob}")

    new_file = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        for seed in parse_seeds(args.seeds):
            if seed in done:
                print(f"seed {seed}: already in {summary_path}, skipped")
                continue
            result = run_one(cubff, language, neighbours, seed, args)
            row = {"seed": seed, **config, **result}
            writer.writerow(row)
            f.flush()
            print(f"seed {seed}: transition_epoch={result['transition_epoch'] or '-'} "
                  f"final_entropy={result['final_entropy']} "
                  f"final_selfrep_fraction={result['final_selfrep_fraction'] or '-'} "
                  f"50%={result['epoch_selfrep_50pct'] or '-'} "
                  f"({result['wall_seconds']} s)", flush=True)


def summarize(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print("no runs")
        return
    n = len(rows)
    crossed = [r for r in rows if r["transition_epoch"]]
    final = [r for r in rows if r["final_entropy"]
             and float(r["final_entropy"]) >= float(r["entropy_threshold"])]
    for label, k in (("first crossing", len(crossed)),
                     ("above threshold at the end", len(final))):
        lo, hi = wilson_interval(k, n)
        print(f"{label:28s}: {k}/{n} = {100 * k / n:.1f}% "
              f"[{100 * lo:.1f}, {100 * hi:.1f}]")
    if crossed:
        epochs = sorted(int(r["transition_epoch"]) for r in crossed)
        print(f"transition epoch: median {np.median(epochs):.0f}, "
              f"min {epochs[0]}, max {epochs[-1]}")
    extinct = sum(r.get("extinct") == "True" for r in rows)
    if extinct:
        print(f"stopped because the seeded replicator died: {extinct}/{n}")
    for f in SELFREP_MILESTONES:
        key = f"epoch_selfrep_{int(f * 100)}pct"
        values = [int(r[key]) for r in rows if r.get(key)]
        if values:
            print(f"selfrep >= {100 * f:4.0f}% of the grid: {len(values)}/{n} "
                  f"runs, median epoch {np.median(values):.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Spatial (2D or 1D) BFF soup runs (paper Section 2.2, "
                    "Fig. 8).")
    parser.add_argument("--summarize", nargs="+", metavar="CSV",
                        help="Only print a summary of the given summary.csv "
                             "files.")
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--out-dir", help="Output directory (required unless "
                                          "--summarize).")
    parser.add_argument("--lang", default="bff_noheads")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=135)
    parser.add_argument("--radius", type=int, default=2,
                        help="Maximum distance along each coordinate between "
                             "two programs that can interact (paper: 2).")
    parser.add_argument("--well-mixed", action="store_true",
                        help="Ignore the grid: standard random pairing of all "
                             "the programs (zero-dimensional reference).")
    parser.add_argument("--max-epochs", type=int, default=16000)
    parser.add_argument("--mutation-prob", type=float, default=2 ** -12,
                        help="Per-byte mutation probability (default 2^-12, "
                             "the engine and paper default).")
    parser.add_argument("--long-range-prob", type=float, default=0.0,
                        help="Probability that a program picks its partner "
                             "among all the programs of the grid instead of "
                             "among its neighbours (needs "
                             "long_range_prob.patch).")
    parser.add_argument("--entropy-threshold", type=float,
                        default=PAPER_TRANSITION_ENTROPY_THRESHOLD)
    parser.add_argument("--callback-interval", type=int, default=64)
    parser.add_argument("--no-selfrep", action="store_true",
                        help="Do not run the engine's self-replication check "
                             "(faster, but no takeover measures).")
    parser.add_argument("--snapshot-interval", type=int, default=1024,
                        help="Epochs between grid snapshots (0 = none). Must "
                             "be a multiple of --callback-interval.")
    parser.add_argument("--stop-fraction", type=float, default=None,
                        help="Stop a run when this fraction of the grid is "
                             "made of self-replicators.")
    parser.add_argument("--stop-if-extinct-after", type=int, default=None,
                        metavar="EPOCH",
                        help="For seeded runs: stop as soon as no program "
                             "is a self-replicator at a measurement made at "
                             "or after EPOCH (the seeded replicator has "
                             "died); the run is marked extinct=True.")
    parser.add_argument("--initial-program", default=None,
                        help="Replace one program of the random initial grid "
                             "('paper_fig4' or 128 hex digits), as in "
                             "transition_survey.py.")
    parser.add_argument("--initial-position", choices=["random", "center"],
                        default="random",
                        help="Where to put --initial-program: a random cell "
                             "(depends on the seed) or the centre of the grid.")
    parser.add_argument("--initial-program-index", type=int, default=None)
    parser.add_argument("--tmp-dir", default=None)
    args = parser.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.out_dir:
        parser.error("--out-dir is required")
    if (args.width * args.height) % 2:
        parser.error("width * height must be even")
    if args.snapshot_interval % args.callback_interval:
        parser.error("--snapshot-interval must be a multiple of "
                     "--callback-interval")
    if not 0 <= args.long_range_prob <= 1:
        parser.error("--long-range-prob must be in [0, 1]")
    if args.stop_if_extinct_after is not None and not args.initial_program:
        parser.error("--stop-if-extinct-after is only for seeded runs")
    if args.well_mixed and args.long_range_prob > 0:
        parser.error("--well-mixed and --long-range-prob are incompatible")
    run_survey(args)


if __name__ == "__main__":
    main()
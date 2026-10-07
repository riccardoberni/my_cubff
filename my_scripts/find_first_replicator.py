#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_first_replicator.py

Second step of the tracer-token case study (paper Section 2.1): use the
tokens to pinpoint the epoch and the tape where the first self-replicator
emerges.

After the state transition the soup is dominated by a few tokens (see
token_soup.py). This script takes the origins of the 32 most frequent tokens
at the first sampled epoch where the high-order entropy is above the
threshold, re-runs the token simulation of the same seed, and in a window of
epochs before the transition counts, for every program, how many of those
origins it holds. Before the first replicator exists they are scattered over
different tapes; the first replicator is the first tape that gathers most of
them, and from then on the number of such tapes grows from epoch to epoch.

For the epoch of emergence the script saves the pair of programs that
interacted (after the mutation step, before execution) with their origin
ids, so that the interaction can be replayed step by step (plot_fig3.py),
and asks the engine for the self-replication score of the two programs
before and after the interaction.

The token simulation is deterministic and was checked against the engine by
token_soup.py for the same seed; as a consistency check, the number of
unique tokens at the end of the window is compared with the one recorded
there.

Usage (from the cubff folder, after running token_soup.py for the seed):
    python3 -u my_scripts/find_first_replicator.py --seed 171
"""

import argparse
import csv
import ctypes
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import token_soup as ts

COMMANDS = b"[]+-.,<>{}"


def tape_text(tape):
    """Commands as themselves, the null byte as '0', anything else blank."""
    return "".join("0" if b == 0 else chr(b) if b in COMMANDS else " "
                   for b in bytes(tape))


def read_dominant_origins(tokens_dir, seed, threshold):
    """Origins of the 32 most frequent tokens at the first sampled epoch
    whose high-order entropy is at least `threshold`."""
    stats_path = os.path.join(tokens_dir, f"seed{seed}_stats.csv")
    top_path = os.path.join(tokens_dir, f"seed{seed}_top32.csv")
    jump_epoch = None
    unique_at = {}
    with open(stats_path, newline="") as f:
        for row in csv.DictReader(f):
            unique_at[int(row["epoch"])] = int(row["unique_tokens"])
            if (jump_epoch is None
                    and float(row["higher_entropy"]) >= threshold):
                jump_epoch = int(row["epoch"])
    if jump_epoch is None:
        sys.exit(f"no sample with high-order entropy >= {threshold} in "
                 f"{stats_path}")
    origins = []
    with open(top_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == jump_epoch:
                origin = ((int(row["origin_epoch"]) << ts.POSITION_BITS)
                          | (int(row["origin_program"]) * ts.TAPE_SIZE
                             + int(row["origin_offset"])))
                if origin not in origins:
                    origins.append(origin)
    return jump_epoch, np.array(origins, dtype=np.uint64), unique_at


def distinct_tracked_per_program(origin, tracked, num_programs):
    """For every program, how many of the tracked origins it holds; also the
    total number of soup bytes carrying a tracked origin."""
    grid = origin.reshape(num_programs, ts.TAPE_SIZE)
    mask = np.isin(grid, tracked)
    rows = np.flatnonzero(mask.any(axis=1))
    distinct = np.zeros(num_programs, dtype=np.int32)
    sub = grid[rows]
    for o in tracked:
        distinct[rows] += (sub == o).any(axis=1)
    return distinct, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Find the epoch and the tape where the first "
                    "self-replicator emerges, using tracer tokens.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tokens-dir",
                        default="soup_inspector_results/tokens",
                        help="Folder with the output of token_soup.py.")
    parser.add_argument("--entropy-threshold", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=256,
                        help="Number of epochs before the transition in "
                             "which programs are examined at every epoch.")
    parser.add_argument("--min-fraction", type=float, default=2 / 3,
                        help="A tape counts as carrying the replicator's "
                             "tokens if it holds at least this fraction of "
                             "the tracked origins.")
    parser.add_argument("--num-programs", type=int, default=131072)
    parser.add_argument("--mutation-prob", type=float, default=2 ** -12)
    parser.add_argument("--lang", default="bff_noheads")
    args = parser.parse_args()

    jump_epoch, tracked, unique_at = read_dominant_origins(
        args.tokens_dir, args.seed, args.entropy_threshold)
    needed = math.ceil(args.min_fraction * len(tracked))
    start = max(0, jump_epoch - args.window)
    print(f"seed {args.seed}: first sample above the threshold at epoch "
          f"{jump_epoch}; tracking {len(tracked)} origins; a tape carries "
          f"the replicator's tokens if it holds >= {needed} of them; "
          f"examining epochs {start + 1}-{jump_epoch}")

    cubff = ts.import_cubff()
    lib = ts.load_token_library()
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.ts_pair_before_execution.argtypes = [
        u8p, u64p, u32p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint32, ctypes.c_uint32, u8p, u64p, u32p, u32p]
    lib.ts_pair_before_execution.restype = None

    mutation_raw = round(args.mutation_prob * (1 << 30))
    n = args.num_programs
    sim = ts.TokenSoup(lib, args.seed, n, mutation_raw)
    while sim.epochs_done < start:
        sim.step()
    print(f"reached epoch {start}")

    table_path = os.path.join(args.tokens_dir,
                              f"seed{args.seed}_emergence.csv")
    found = None
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tracked_bytes", "max_tracked_on_one_tape",
                         "tapes_with_replicator_tokens"])
        distinct, _ = distinct_tracked_per_program(sim.origin, tracked, n)
        if distinct.max() >= needed:
            sys.exit("a tape already carries the replicator's tokens at the "
                     "start of the window: increase --window")
        while sim.epochs_done < jump_epoch:
            if found is None:
                soup_before = sim.soup.copy()
                origin_before = sim.origin.copy()
            loop_epoch = sim.epochs_done  # epoch index used by the engine
            sim.step()
            distinct, total = distinct_tracked_per_program(
                sim.origin, tracked, n)
            carriers = int((distinct >= needed).sum())
            writer.writerow([sim.epochs_done, total, int(distinct.max()),
                             carriers])
            if found is None and carriers > 0:
                program = int(distinct.argmax())
                tape = np.zeros(2 * ts.TAPE_SIZE, dtype=np.uint8)
                tok = np.zeros(2 * ts.TAPE_SIZE, dtype=np.uint64)
                p1, p2 = ctypes.c_uint32(), ctypes.c_uint32()
                lib.ts_pair_before_execution(
                    soup_before.ctypes.data_as(u8p),
                    origin_before.ctypes.data_as(u64p),
                    sim.pairing.ctypes.data_as(u32p), n, args.seed,
                    loop_epoch, mutation_raw, program,
                    tape.ctypes.data_as(u8p), tok.ctypes.data_as(u64p),
                    ctypes.byref(p1), ctypes.byref(p2))
                first, second = p1.value, p2.value
                after = np.concatenate([
                    sim.soup[first * ts.TAPE_SIZE:(first + 1) * ts.TAPE_SIZE],
                    sim.soup[second * ts.TAPE_SIZE:
                             (second + 1) * ts.TAPE_SIZE]])
                found = {"loop_epoch": loop_epoch, "program": program,
                         "p1": first, "p2": second, "tape": tape, "tok": tok,
                         "after": after,
                         "distinct": int(distinct[program])}
                del soup_before, origin_before
    if found is None:
        sys.exit("no tape gathers the tracked origins inside the window")

    # Consistency with the engine-checked run of token_soup.py.
    keys, _ = sim.token_counts()
    expected = unique_at.get(sim.epochs_done)
    print(f"unique tokens at epoch {sim.epochs_done}: {len(keys)} "
          f"(token_soup.py recorded {expected}) -> "
          f"{'consistent' if len(keys) == expected else 'NOT CONSISTENT'}")

    # Self-replication score of the two programs before and after the
    # interaction, computed by the engine (0-64).
    language = cubff.GetLanguage(args.lang)
    threshold = getattr(cubff, "kSelfrepThreshold", None)

    def score(program_bytes):
        return int(language.EvalParsedSelfrep(
            cubff.VectorUint8(bytes(program_bytes)), found["loop_epoch"],
            args.seed, False))

    half = ts.TAPE_SIZE
    before, after = found["tape"], found["after"]
    scores = {"A": score(before[:half]), "B": score(before[half:]),
              "A_after": score(after[:half]), "B_after": score(after[half:])}

    out_path = os.path.join(args.tokens_dir,
                            f"seed{args.seed}_first_replicator.npz")
    np.savez(out_path, seed=args.seed, loop_epoch=found["loop_epoch"],
             program_a=found["p1"], program_b=found["p2"],
             tape=found["tape"], origin=found["tok"], after=found["after"],
             tracked=tracked)

    print(f"\nfirst replicator: created by the interaction of epoch index "
          f"{found['loop_epoch']} (the soup after {found['loop_epoch'] + 1} "
          f"epochs contains it)")
    print(f"pair: program {found['p1']} (tape A, executed first) + program "
          f"{found['p2']} (tape B)")
    print(f"tape holding the tracked tokens afterwards: program "
          f"{found['program']} ({found['distinct']} of {len(tracked)} "
          f"tracked origins)")
    print(f"A  before: |{tape_text(before[:half])}|")
    print(f"B  before: |{tape_text(before[half:])}|")
    print(f"A  after:  |{tape_text(after[:half])}|")
    print(f"B  after:  |{tape_text(after[half:])}|")
    print(f"self-replication score (engine, 0-64"
          + (f", threshold {threshold}" if threshold is not None else "")
          + f"): A {scores['A']}, B {scores['B']} before; "
          f"A {scores['A_after']}, B {scores['B_after']} after")
    print(f"written {table_path} and {out_path}")

    print("\ntapes carrying the replicator's tokens, epoch by epoch:")
    with open(table_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if int(r["epoch"]) >= found["loop_epoch"]]
    print(" ".join(f"{r['epoch']}:{r['tapes_with_replicator_tokens']}"
                   for r in rows[:40]))


if __name__ == "__main__":
    main()
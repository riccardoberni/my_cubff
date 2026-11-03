#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
longtape_patterns.py

Experiment 7: search for self-replicators on the long tape, from the tape
snapshots written by long_tape_bff.py --dump-tape-every
(<results-dir>/seed<N>/tape_dumps_seed<N>/tape_<instructions>.bin).

On the long tape there are no programs with fixed boundaries, so the
self-replication check of the engine cannot be applied. The analysis looks
instead for structure that copying would leave on the tape, and then
executes the tape again from positions inside that structure to see whether
the execution copies it further.

For every seed:
  1. byte counts: for every snapshot, the three most common byte values and
     the rank of each BFF command among all byte values
     (byte_counts.csv);
  2. dominated regions: in two snapshots, the first one taken after the
     high-order entropy first exceeds the threshold and the last one within
     --max-instructions, the tape is divided into windows of --window bytes;
     a window is dominated by a byte value if that value fills at least
     --min-share of it, and consecutive windows dominated by the same value
     are merged into a region (regions.csv);
  3. repeated substrings: in the same snapshots, the substrings of --k bytes
     that occur at least --min-repeats times and contain at least two
     different byte values (on a random tape of 65536 bytes the expected
     number of repeated 8-byte substrings is about 1e-10); the fraction of
     the tape they cover is reported (repeats.csv);
  4. re-execution: from --candidates positions inside the dominated regions
     and the repeated substrings, and from as many uniformly random positions
     (baseline), one execution window is run on a copy of the snapshot,
     exactly as in long_tape_bff.py (same step limit, head1 offset and
     wrap-around), recording every copy instruction. A copy is "new" if it
     changes the destination byte and "redundant" if the destination already
     held the copied value. For each group the script reports the fraction
     of executions with at least one new copy, the mean numbers of new and
     redundant copies, and the mean change of the tape coverage of the
     repeated substrings (reexecution.csv).

Everything is deterministic: the positions are drawn from
numpy.random.default_rng([seed, instructions]).

Usage (from the repository root):
    python3 -u my_scripts/longtape_patterns.py --results-dir longtape_results \
        --out-dir longtape_results/patterns
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from long_tape_bff import BffOp, get_op_kind, run_one_window  # noqa: E402

COMMANDS = "[]+-.,<>{}"


def symbol(b):
    if b == 0:
        return "0"
    if chr(b) in COMMANDS:
        return chr(b)
    if 33 <= b < 127:
        return chr(b)
    return f"\\x{b:02x}"


def list_dumps(seed_dir):
    """(instructions, path) of the .bin snapshots of one seed, sorted."""
    out = []
    for path in glob.glob(os.path.join(seed_dir, "tape_dumps_seed*",
                                       "tape_*.bin")):
        m = re.search(r"tape_(\d+)\.bin$", path)
        out.append((int(m.group(1)), path))
    return sorted(out)


def first_crossing(seed_dir, max_instructions, threshold):
    for path in sorted(glob.glob(os.path.join(seed_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["instructions"] > max_instructions:
                    break
                if r["higher_entropy"] > threshold:
                    return r["instructions"]
    return None


def byte_rows(seed, instructions, tape):
    counts = Counter(tape)
    ranking = [b for b, _ in counts.most_common()]
    rows = []
    for rank, b in enumerate(ranking[:3], 1):
        rows.append({"seed": seed, "instructions": instructions,
                     "kind": "top", "rank": rank, "byte": b,
                     "symbol": symbol(b), "count": counts[b]})
    for c in COMMANDS:
        b = ord(c)
        rank = ranking.index(b) + 1 if b in counts else ""
        rows.append({"seed": seed, "instructions": instructions,
                     "kind": "command", "rank": rank, "byte": b,
                     "symbol": c, "count": counts.get(b, 0)})
    return rows


def dominated_regions(tape, window, min_share):
    """Regions of consecutive windows dominated by the same byte value:
    list of (start, length, byte, share of that byte in the region)."""
    arr = np.frombuffer(bytes(tape), dtype=np.uint8)
    n = len(arr) // window
    dominant = []
    for i in range(n):
        values, counts = np.unique(arr[i * window:(i + 1) * window],
                                   return_counts=True)
        j = counts.argmax()
        dominant.append(int(values[j]) if counts[j] >= min_share * window
                        else None)
    regions = []
    i = 0
    while i < n:
        if dominant[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and dominant[j + 1] == dominant[i]:
            j += 1
        start, length = i * window, (j - i + 1) * window
        share = float((arr[start:start + length] == dominant[i]).mean())
        regions.append((start, length, dominant[i], share))
        i = j + 1
    return regions


def repeated_substrings(tape, k, min_repeats):
    """Substrings of k bytes (no wrap-around) occurring at least min_repeats
    times and made of at least two byte values: {substring: [positions]}."""
    data = bytes(tape)
    positions = {}
    for i in range(len(data) - k + 1):
        positions.setdefault(data[i:i + k], []).append(i)
    return {s: p for s, p in positions.items()
            if len(p) >= min_repeats and len(set(s)) >= 2}


def coverage(tape, substrings, k):
    """Number of tape positions covered by an occurrence of any of the
    given substrings."""
    data = bytes(tape)
    covered = np.zeros(len(data) + k, dtype=np.int32)
    for i in range(len(data) - k + 1):
        if data[i:i + k] in substrings:
            covered[i] += 1
            covered[i + k] -= 1
    return int((np.cumsum(covered)[:len(data)] > 0).sum())


def traced_window(tape, tape_size, pc0, head1_offset, max_steps):
    """Same execution as long_tape_bff.run_one_window, recording the copy
    instructions. Returns (executed steps, new copies, redundant copies)."""
    pc = pc0 % tape_size
    head0 = pc0 % tape_size
    head1 = (pc0 + head1_offset) % tape_size
    executed = new = redundant = 0
    for _ in range(max_steps):
        kind = get_op_kind(tape[pc])
        if kind == BffOp.DEC0:
            head0 = (head0 - 1) % tape_size
        elif kind == BffOp.INC0:
            head0 = (head0 + 1) % tape_size
        elif kind == BffOp.DEC1:
            head1 = (head1 - 1) % tape_size
        elif kind == BffOp.INC1:
            head1 = (head1 + 1) % tape_size
        elif kind == BffOp.PLUS:
            tape[head0] = (tape[head0] + 1) & 0xFF
        elif kind == BffOp.MINUS:
            tape[head0] = (tape[head0] - 1) & 0xFF
        elif kind == BffOp.COPY01:
            if tape[head1] == tape[head0]:
                redundant += 1
            else:
                new += 1
            tape[head1] = tape[head0]
        elif kind == BffOp.COPY10:
            if tape[head0] == tape[head1]:
                redundant += 1
            else:
                new += 1
            tape[head0] = tape[head1]
        elif kind == BffOp.LOOP_START:
            if get_op_kind(tape[head0]) == BffOp.NULL:
                depth, scan, found = 1, pc + 1, False
                for _ in range(tape_size):
                    c = get_op_kind(tape[scan % tape_size])
                    if c == BffOp.LOOP_END:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    elif c == BffOp.LOOP_START:
                        depth += 1
                    scan += 1
                if not found:
                    break
                pc = scan
        elif kind == BffOp.LOOP_END:
            if get_op_kind(tape[head0]) != BffOp.NULL:
                depth, scan, found = 1, pc - 1, False
                for _ in range(tape_size):
                    c = get_op_kind(tape[scan % tape_size])
                    if c == BffOp.LOOP_END:
                        depth += 1
                    elif c == BffOp.LOOP_START:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    scan -= 1
                if not found:
                    break
                pc = scan
        executed += 1
        pc = (pc + 1) % tape_size
    return executed, new, redundant


def check_traced_window(tape, rng, head1_offset, max_steps, n=20):
    """The traced execution must leave the tape exactly as run_one_window."""
    for pc0 in rng.integers(0, len(tape), n):
        a, b = bytearray(tape), bytearray(tape)
        ra = run_one_window(a, len(a), int(pc0), head1_offset, max_steps)
        executed, _, _ = traced_window(b, len(b), int(pc0), head1_offset,
                                       max_steps)
        if a != b or ra.instructions_executed != executed:
            sys.exit("traced execution differs from run_one_window")


def reexecute(tape, positions, substrings, k, head1_offset, max_steps):
    base = coverage(tape, substrings, k)
    rows = []
    for pc0 in positions:
        t = bytearray(tape)
        executed, new, redundant = traced_window(t, len(t), int(pc0),
                                                 head1_offset, max_steps)
        rows.append((executed, new, redundant,
                     coverage(t, substrings, k) - base))
    return rows


def summarize(rows):
    a = np.array(rows, dtype=float).reshape(-1, 4)
    return {"executions": len(a),
            "with_new_copies": float((a[:, 1] > 0).mean()),
            "mean_steps": float(a[:, 0].mean()),
            "mean_new_copies": float(a[:, 1].mean()),
            "mean_redundant_copies": float(a[:, 2].mean()),
            "mean_coverage_change": float(a[:, 3].mean()),
            "max_coverage_change": float(a[:, 3].max())}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="longtape_results")
    parser.add_argument("--out-dir", default=None,
                        help="default: <results-dir>/patterns")
    parser.add_argument("--max-instructions", type=int,
                        default=2_000_000_000)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--min-share", type=float, default=0.25)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--min-repeats", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=500)
    parser.add_argument("--head1-offset", type=int, default=16)
    parser.add_argument("--max-steps-per-window", type=int, default=8192)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "patterns")
    os.makedirs(out_dir, exist_ok=True)

    seed_dirs = sorted((d for d in glob.glob(os.path.join(args.results_dir,
                                                        "seed*"))
                        if os.path.isdir(d)),
                       key=lambda d: int(re.sub(r"\D", "", os.path.basename(d))))
    byte_out, region_out, repeat_out, exec_out = [], [], [], []
    for seed_dir in seed_dirs:
        seed = int(re.sub(r"\D", "", os.path.basename(seed_dir)))
        # The last snapshot of a run is taken just after --max-instructions
        # (a window is never interrupted), hence the tolerance of one window.
        dumps = [(i, p) for i, p in list_dumps(seed_dir)
                 if i <= args.max_instructions + args.max_steps_per_window]
        if not dumps:
            continue
        tapes = {}
        for instructions, path in dumps:
            with open(path, "rb") as f:
                tapes[instructions] = bytearray(f.read())
            byte_out += byte_rows(seed, instructions, tapes[instructions])

        crossing = first_crossing(seed_dir, args.max_instructions,
                                  args.threshold)
        chosen = [dumps[-1][0]]
        if crossing is not None:
            after = [i for i, _ in dumps if i >= crossing]
            if after and after[0] != chosen[0]:
                chosen.insert(0, after[0])
        print(f"seed {seed}: {len(dumps)} snapshots, first crossing "
              f"{crossing if crossing is not None else '-'}, analysed "
              f"{', '.join(f'{c:,}' for c in chosen)}")

        for instructions in chosen:
            tape = tapes[instructions]
            rng = np.random.default_rng([seed, instructions])
            check_traced_window(tape, rng, args.head1_offset,
                                args.max_steps_per_window)

            regions = dominated_regions(tape, args.window, args.min_share)
            for start, length, b, share in regions:
                region_out.append({"seed": seed, "instructions": instructions,
                                   "start": start, "length": length,
                                   "byte": b, "symbol": symbol(b),
                                   "share": f"{share:.3f}"})
            longest = max(regions, key=lambda r: r[1]) if regions else None

            repeats = repeated_substrings(tape, args.k, args.min_repeats)
            covered = coverage(tape, repeats, args.k)
            top = sorted(repeats.items(), key=lambda kv: -len(kv[1]))[:10]
            for s, pos in top:
                repeat_out.append({
                    "seed": seed, "instructions": instructions,
                    "substring": "".join(symbol(b) for b in s),
                    "occurrences": len(pos),
                    "commands": sum(chr(b) in COMMANDS for b in s),
                    "first_position": pos[0]})

            inside = set()
            for start, length, _, _ in regions:
                inside.update(range(start, start + length))
            for pos in repeats.values():
                for p in pos:
                    inside.update(range(p, p + args.k))
            inside = np.array(sorted(inside), dtype=np.int64)
            n = min(args.candidates, len(inside))
            groups = {"random": rng.integers(0, len(tape), args.candidates)}
            if n:
                groups["structure"] = rng.choice(inside, n, replace=False)
            for name, positions in groups.items():
                s = summarize(reexecute(tape, positions, repeats, args.k,
                                        args.head1_offset,
                                        args.max_steps_per_window))
                exec_out.append({"seed": seed, "instructions": instructions,
                                 "group": name,
                                 **{key: (f"{v:.4f}" if isinstance(v, float)
                                          else v) for key, v in s.items()}})
            print(f"  {instructions:>13,}: {len(regions)} dominated regions"
                  + (f" (longest {longest[1]} bytes of "
                     f"'{symbol(longest[2])}', share {longest[3]:.2f})"
                     if longest else "")
                  + f"; {len(repeats)} repeated {args.k}-byte substrings "
                    f"covering {covered} bytes")
            for row in exec_out[-len(groups):]:
                print(f"     re-execution {row['group']:>9}: "
                      f"{row['executions']} runs, new copies in "
                      f"{100 * float(row['with_new_copies']):.1f}%, "
                      f"mean new {row['mean_new_copies']}, redundant "
                      f"{row['mean_redundant_copies']}, coverage change "
                      f"{row['mean_coverage_change']} "
                      f"(max {row['max_coverage_change']})")

    for name, rows in (("byte_counts.csv", byte_out),
                       ("regions.csv", region_out),
                       ("repeats.csv", repeat_out),
                       ("reexecution.csv", exec_out)):
        if rows:
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"tables written to {out_dir}")


if __name__ == "__main__":
    main()

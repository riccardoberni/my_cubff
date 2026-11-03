#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_longtape.py

Long-tape runs (Experiment 6): reading, transition detection and figure.

long_tape_bff.py appends one JSON record per measurement to
<output-dir>/seed<N>/seed<N>_<instr>instr.jsonl, with the keys
instructions, windows, higher_entropy, h0, brotli_bpb (compressed bits per
byte, zlib when brotli is not installed), top1_freq, top16_freq, mops_s.
This module reads those files, finds the first measurement at which the
high-order entropy exceeds the threshold, and draws the figure of one run:
high-order and Shannon entropy (top) and the fraction of the tape occupied by
the most common byte and by the 16 most common bytes (bottom), against the
number of executed instructions.

Usage (from the repository root):
    python3 my_scripts/plot_longtape.py \
        --jsonl my_scripts/longtape_results/seed0/seed0_2000000000instr.jsonl \
        --out images/longtape_seed0.pdf
With several --jsonl files it also prints, for each run, the first crossing,
the maximum and the final value of the high-order entropy.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD = 1.0


def save_records(records, path):
    """Append records to a JSON-lines file."""
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def first_crossing(records, threshold=THRESHOLD, x_key="instructions"):
    """First value of x_key at which the high-order entropy exceeds the
    threshold, or None."""
    for r in records:
        h = r.get("higher_entropy")
        if h is not None and not math.isnan(h) and h > threshold:
            return r[x_key]
    return None


def plot_longtape(records, out, threshold=THRESHOLD):
    x = [r["instructions"] for r in records]
    t = first_crossing(records, threshold)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)

    ax1.plot(x, [r["higher_entropy"] for r in records], color="crimson",
             label="high-order entropy")
    ax1.plot(x, [r["h0"] for r in records], color="steelblue", alpha=0.8,
             label="Shannon entropy")
    ax1.axhline(threshold, color="grey", linestyle=":", linewidth=1,
                label="threshold")
    ax1.set_ylabel("bits per byte")
    ax1.legend(loc="center left", fontsize=8)

    ax2.plot(x, [r["top1_freq"] for r in records], color="darkorange",
             label="most common byte")
    ax2.plot(x, [r["top16_freq"] for r in records], color="olive",
             label="16 most common bytes")
    ax2.set_ylabel("fraction of the tape")
    ax2.set_ylim(0, 0.3)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("instructions executed")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        if t is not None:
            ax.axvline(t, color="black", linestyle="--", linewidth=1,
                       alpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", nargs="+", required=True,
                        help="Traces written by long_tape_bff.py; the "
                             "figure is drawn for the first one.")
    parser.add_argument("--out", default=None,
                        help="Output figure (PDF or PNG).")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    for path in args.jsonl:
        records = load_records(path)
        h = [r["higher_entropy"] for r in records]
        t = first_crossing(records, args.threshold)
        print(f"{os.path.basename(path)}: first crossing "
              f"{'-' if t is None else f'{t:,}'}, max {max(h):.3f}, "
              f"final {h[-1]:.3f} at {records[-1]['instructions']:,} "
              f"instructions")
    if args.out:
        plot_longtape(load_records(args.jsonl[0]), args.out, args.threshold)
        print(f"figure written to {args.out}")


if __name__ == "__main__":
    main()

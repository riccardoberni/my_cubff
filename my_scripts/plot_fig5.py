#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig5.py

Reproduces Fig. 5 of the paper: distribution of the soup's high-order
entropy over time across many runs, drawn as quantile ranges in 5% steps
shaded symmetrically around the median, on the same log(1 + x) vertical
scale as the paper's figure.

Input: the per-seed entropy traces written by
    transition_survey.py --full-run --trace-dir <dir>
i.e. files <dir>/seed<N>.csv with columns "epoch,higher_entropy".

Usage (from ~/cubff):
    python3 my_scripts/plot_fig5.py \
        --trace-dir soup_inspector_results/traces_fig5 \
        --out soup_inspector_results/fig5_ours.pdf

Besides the figure, it prints the numbers needed to compare with the
paper's text: number of runs, fraction of runs whose entropy ever reaches
the threshold, fraction above the threshold at the end of the run, and the
median entropy at a few reference epochs (the paper notes an increase over
the first ~1000 epochs followed by a decrease).
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # no display on cluster login/compute nodes
import matplotlib.pyplot as plt


def read_trace(path):
    epochs, values = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            values.append(float(row["higher_entropy"]))
    return np.array(epochs), np.array(values)


def load_traces(trace_dir, max_epoch):
    paths = glob.glob(os.path.join(trace_dir, "seed*.csv"))
    if not paths:
        sys.exit(f"No seed*.csv files found in {trace_dir}")

    def seed_of(p):
        m = re.search(r"seed(\d+)\.csv$", p)
        return int(m.group(1)) if m else -1

    paths.sort(key=seed_of)
    traces = {}
    for p in paths:
        e, v = read_trace(p)
        if len(e) == 0:
            print(f"WARNING: empty trace {p}, skipped")
            continue
        traces[seed_of(p)] = (e, v)

    # Keep only complete runs: an interrupted run (e.g. killed by a SLURM
    # timeout and later re-run) would otherwise bias the late-epoch bands.
    complete = {s: t for s, t in traces.items() if t[0][-1] >= max_epoch}
    dropped = sorted(set(traces) - set(complete))
    if dropped:
        print(f"WARNING: {len(dropped)} incomplete trace(s) dropped "
              f"(last epoch < {max_epoch}): seeds {dropped}")
    if not complete:
        sys.exit("No complete traces left.")

    # All complete runs share the same callback grid; check it explicitly.
    ref = next(iter(complete.values()))[0]
    for s, (e, _) in complete.items():
        if len(e) != len(ref) or not np.array_equal(e, ref):
            sys.exit(f"Seed {s} has a different epoch grid; aborting.")
    seeds = sorted(complete)
    matrix = np.stack([complete[s][1] for s in seeds])  # (runs, epochs)
    return seeds, ref, matrix


def plot_bands(epochs, matrix, out):
    """Same layout as the paper's Fig. 5, so that the two figures can be
    compared directly:
      * quantile ranges in 5% steps, shaded symmetrically around the median
        (darkest for 45-55%, lightest for 0-5% and 95-100%);
      * vertical axis proportional to log(1 + entropy), which is the scale
        used in the paper's figure (its ticks 0..7 are spaced that way);
      * horizontal colour legend below the plot.
    """
    qs = np.arange(0, 101, 5)  # 0, 5, ..., 100 percentiles
    quant = np.percentile(matrix, qs, axis=0)  # (21, epochs)

    # Bands: 0-5, ..., 40-45, 45-55, 55-60, ..., 95-100 (19 bands; the
    # central one spans 10%, as in the paper's legend).
    edges = [q for q in qs if q != 50]
    light = np.array([1.0, 0.84, 0.85])
    dark = np.array([1.0, 0.0, 0.0])

    def shade(lo, hi):
        depth = 1 - abs((lo + hi) / 2 - 50) / 47.5  # 0 at the tails, 1 at 50%
        return tuple(light + (dark - light) * depth)

    bands = [(lo, hi, shade(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]

    fig, (ax, cax) = plt.subplots(
        2, 1, figsize=(10, 5.6),
        gridspec_kw={"height_ratios": [14, 1], "hspace": 0.32})
    for lo, hi, color in bands:
        ax.fill_between(epochs, quant[lo // 5], quant[hi // 5],
                        color=color, linewidth=0)

    ax.set_yscale("function", functions=(np.log1p, np.expm1))
    ax.set_ylim(0, 7)
    ax.set_yticks(range(8))
    ax.set_xlim(0, 16384)
    ax.set_xticks(range(0, 16001, 1000))
    ax.grid(True, color="0.6", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("high-order entropy")

    # Colour legend: one patch per band, at its true width.
    for lo, hi, color in bands:
        cax.axvspan(lo, hi, color=color, linewidth=0)
    cax.set_xlim(0, 100)
    cax.set_yticks([])
    cax.set_xticks(range(0, 101, 10))
    cax.set_xticklabels([f"{q}%" for q in range(0, 101, 10)])

    fig.savefig(out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(out)[0] + ".png"
    if png != out:
        fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Figure written to {out}" + (f" and {png}" if png != out else ""))


def print_summary(seeds, epochs, matrix, threshold):
    n = matrix.shape[0]
    ever = (matrix >= threshold).any(axis=1)
    final = matrix[:, -1] >= threshold
    print(f"\nruns: {n} (seeds {seeds[0]}-{seeds[-1]})")
    print(f"ever reached threshold (first-crossing criterion): "
          f"{ever.sum()}/{n} = {ever.mean():.1%}")
    print(f"above threshold at the end (end-of-run criterion):  "
          f"{final.sum()}/{n} = {final.mean():.1%}")
    print(f"crossed and fell back below by the end: "
          f"{(ever & ~final).sum()}")

    median = np.median(matrix, axis=0)
    # Restrict the early-peak search to the pre-transition regime the paper
    # describes (first ~2000 epochs).
    early = epochs <= 2000
    k = int(np.argmax(median[early]))
    print(f"\nmedian entropy at first recorded epoch {epochs[0]}: "
          f"{median[0]:.3f}")
    print(f"max of median over epochs <= 2000: {median[early][k]:.3f} "
          f"at epoch {epochs[early][k]}")
    for target in (1000, 2000, 4000, 8000, 16000):
        j = int(np.argmin(np.abs(epochs - target)))
        print(f"median at epoch {epochs[j]:>5}: {median[j]:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantile bands of high-order entropy over time "
                    "(paper Fig. 5).")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", default="fig5_ours.pdf")
    parser.add_argument("--max-epoch", type=int, default=16000,
                        help="A trace counts as complete if its last "
                             "recorded epoch is >= this value.")
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    seeds, epochs, matrix = load_traces(args.trace_dir, args.max_epoch)
    plot_bands(epochs, matrix, args.out)
    print_summary(seeds, epochs, matrix, args.threshold)


if __name__ == "__main__":
    main()
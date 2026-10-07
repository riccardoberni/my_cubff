#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig1.py

Reproduces Fig. 1 of the paper from the statistics written by token_soup.py:
number of unique tokens and number of soup bytes holding one of the 32 most
frequent tokens (left axis, logarithmic), together with the high-order
entropy of the soup (right axis), as a function of the epoch. As in the
paper, the number of unique tokens of a run without a state transition is
shown for comparison.

Usage (from the cubff folder):
    python3 my_scripts/plot_fig1.py \
        --stats soup_inspector_results/tokens/seed171_stats.csv \
        --reference-stats soup_inspector_results/tokens/seed1_stats.csv \
        --out soup_inspector_results/fig1_ours.pdf
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_stats(path):
    columns = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    if not columns:
        raise SystemExit(f"{path} is empty")
    if min(columns["soup_matches_engine"]) < 1:
        print(f"WARNING: {path} contains samples where the token simulation "
              f"differs from the engine")
    return columns


def main():
    parser = argparse.ArgumentParser(
        description="Unique tokens, top-32 tokens and complexity over time "
                    "(paper Fig. 1).")
    parser.add_argument("--stats", required=True,
                        help="token_soup.py statistics of the run with a "
                             "state transition.")
    parser.add_argument("--reference-stats", default=None,
                        help="Statistics of a run without transition "
                             "(grey line).")
    parser.add_argument("--first-replicator", type=int, default=None,
                        help="Epoch at which the first replicator emerges "
                             "(dashed vertical line).")
    parser.add_argument("--max-epoch", type=int, default=5000)
    parser.add_argument("--out", default="fig1_ours.pdf")
    args = parser.parse_args()

    run = read_stats(args.stats)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(run["epoch"], run["unique_tokens"], color="blue", linewidth=1.0,
            label="unique tokens")
    if args.reference_stats:
        ref = read_stats(args.reference_stats)
        ax.plot(ref["epoch"], ref["unique_tokens"], color="0.55",
                linewidth=1.0, label="unique (no transition)")
    ax.plot(run["epoch"], run["top32_tokens"], color="red", linewidth=1.0,
            label="top32 tokens")
    ax.set_yscale("log")
    ax.set_xlim(0, args.max_epoch)
    ax.set_xticks(range(0, args.max_epoch + 1, 500))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Element count")
    ax.grid(True, which="major", color="0.6", linewidth=0.4, alpha=0.6)
    ax.grid(True, which="minor", color="0.8", linewidth=0.3, alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(run["epoch"], run["higher_entropy"], color="green",
             linewidth=1.0, label="complexity")
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("complexity")

    if args.first_replicator is not None:
        ax.axvline(args.first_replicator, color="red", linestyle="--",
                   linewidth=1.0)
        ax.text(args.first_replicator, 0.5, "first replicator ",
                rotation=90, ha="right", va="center",
                transform=ax.get_xaxis_transform(), fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="center left",
              fontsize=9, framealpha=0.9)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    if png != args.out:
        fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Figure written to {args.out}"
          + (f" and {png}" if png != args.out else ""))

    # Numbers for the text.
    last = len(run["epoch"]) - 1
    print(f"\nsamples: {len(run['epoch'])}, epochs "
          f"{int(run['epoch'][0])}-{int(run['epoch'][last])}")
    print(f"unique tokens: {int(run['unique_tokens'][0])} at the first "
          f"sample, {int(min(run['unique_tokens']))} at the minimum, "
          f"{int(run['unique_tokens'][last])} at the end")
    print(f"top32 tokens: {int(run['top32_tokens'][0])} at the first sample, "
          f"{int(max(run['top32_tokens']))} at the maximum")
    print(f"zero bytes: {int(run['zeros'][0])} at the first sample, "
          f"{int(max(run['zeros']))} at the maximum (epoch "
          f"{int(run['epoch'][run['zeros'].index(max(run['zeros']))])})")
    if args.reference_stats:
        print(f"reference run, unique tokens at the end: "
              f"{int(ref['unique_tokens'][-1])}")


if __name__ == "__main__":
    main()
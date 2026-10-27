#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spatial_spread.py

Spread speed of a self-replicator in the spatial soup and in the well-mixed
soup (Experiment 8), from the runs of cluster/run_spatial_spread.sbatch.

The paper (Section 2.2) states that, once a self-replicator appears, it takes
over half of a well-mixed soup of n programs in about log n epochs, and half of
a square 2D grid in a number of epochs proportional to its side, sqrt(n). For
every configuration this script takes the runs in which the seeded replicator
survived and the epoch at which self-replicators first occupied 50% of the
programs (T50, column epoch_selfrep_50pct of summary.csv), and:
  * prints a table: programs, runs, survivors, median and quartiles of T50
    (and of T90);
  * fits T50 = a + b * sqrt(n) and T50 = a + b * log2(n) to the median T50
    of the configurations of each kind of soup (medians, because a few runs
    are much slower than the others), and prints the root-mean-square error
    of both models, so that the model that describes each kind of soup can be
    compared;
  * draws a two-panel figure: median T50 (bars: interquartile range) against
    sqrt(n) and against log2(n), with the fitted model of each kind of soup.

Usage (from the repository root):
    python3 my_scripts/plot_spatial_spread.py --results-dir spatial_results \
        --out images/spatial_spread.pdf
"""

import argparse
import csv
import glob
import math
import os
import re

import numpy as np

GRID_COLOR = "#2a78d6"
MIXED_COLOR = "#eb6834"
INK = "#2b2b2b"
MUTED = "#6b6b6b"


def read_config(directory):
    with open(os.path.join(directory, "summary.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    n = int(rows[0]["width"]) * int(rows[0]["height"])
    t50 = [int(r["epoch_selfrep_50pct"]) for r in rows
           if r["epoch_selfrep_50pct"]]
    t90 = [int(r["epoch_selfrep_90pct"]) for r in rows
           if r["epoch_selfrep_90pct"]]
    return {"n": n, "runs": len(rows), "t50": np.array(t50),
            "t90": np.array(t90)}


def collect(results_dir):
    grids, mixed = [], []
    for d in sorted(glob.glob(os.path.join(results_dir, "spread_grid_*x*"))):
        if re.search(r"spread_grid_\d+x\d+$", d.rstrip("/")):
            c = read_config(d)
            if c:
                grids.append(c)
    for d in sorted(glob.glob(os.path.join(results_dir, "spread_wellmixed_*"))):
        c = read_config(d)
        if c:
            mixed.append(c)
    grids.sort(key=lambda c: c["n"])
    mixed.sort(key=lambda c: c["n"])
    return grids, mixed


def fit(configs, transform):
    """Least-squares fit of median T50 = a + b * transform(n) over the
    configurations. Returns (a, b, rmse)."""
    configs = [c for c in configs if len(c["t50"])]
    x = np.array([transform(c["n"]) for c in configs], dtype=float)
    y = np.array([np.median(c["t50"]) for c in configs], dtype=float)
    if len(x) < 2:
        return None
    b, a = np.polyfit(x, y, 1)
    rmse = float(np.sqrt(np.mean((y - (a + b * x)) ** 2)))
    return a, b, rmse


def quartiles(values):
    if len(values) == 0:
        return (float("nan"),) * 3
    return tuple(np.percentile(values, [25, 50, 75]))


def print_table(label, configs):
    print(f"\n{label}")
    print(f"{'n':>7} {'runs':>5} {'surv.':>6} {'%':>6} "
          f"{'T50 med':>8} {'[q25, q75]':>13} {'T90 med':>8}")
    for c in configs:
        q25, med, q75 = quartiles(c["t50"])
        t90 = np.median(c["t90"]) if len(c["t90"]) else float("nan")
        print(f"{c['n']:>7} {c['runs']:>5} {len(c['t50']):>6} "
              f"{100 * len(c['t50']) / c['runs']:>5.1f}% {med:>8.0f} "
              f"[{q25:>4.0f}, {q75:>4.0f}] {t90:>8.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="spatial_results")
    parser.add_argument("--out", default="images/spatial_spread.pdf")
    args = parser.parse_args()

    grids, mixed = collect(args.results_dir)
    if not grids and not mixed:
        raise SystemExit(f"no spread_* results in {args.results_dir}")
    print_table("2D grid (replicator at the centre)", grids)
    print_table("Well-mixed soup", mixed)

    models = {"sqrt(n)": math.sqrt, "log2(n)": math.log2}
    fits = {}
    print("\nFits median T50 = a + b * f(n) over the configurations "
          "(rmse in epochs):")
    for label, configs in (("2D grid", grids), ("well-mixed", mixed)):
        for name, f in models.items():
            r = fit(configs, f) if configs else None
            fits[(label, name)] = r
            if r:
                print(f"  {label:10s} f = {name:8s}: a = {r[0]:7.1f}, "
                      f"b = {r[1]:7.2f}, rmse = {r[2]:6.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    panels = [("sqrt(n)", math.sqrt, r"$\sqrt{n}$ (side of the grid)"),
              ("log2(n)", math.log2, r"$\log_2 n$")]
    for ax, (name, f, xlabel) in zip(axes, panels):
        for label, configs, color, marker in (
                ("2D grid", grids, GRID_COLOR, "o"),
                ("well-mixed", mixed, MIXED_COLOR, "s")):
            if not configs:
                continue
            x = np.array([f(c["n"]) for c in configs])
            q = np.array([quartiles(c["t50"]) for c in configs])
            ax.errorbar(x, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                        fmt=marker, color=color, ms=6, lw=1.5, capsize=3,
                        label=label, zorder=3)
            r = fits.get((label, name))
            matching = (label == "2D grid") == (name == "sqrt(n)")
            if r and matching:
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, r[0] + r[1] * xs, "--", color=color, lw=1.5,
                        zorder=2, label=f"fit {label}")
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel("epochs to 50% of the programs", color=INK)
        ax.grid(True, color="#e4e4e4", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.set_ylim(bottom=0)
    axes[0].set_title("(a) against the side of the grid", fontsize=10,
                      color=INK)
    axes[1].set_title("(b) against the logarithm of the size", fontsize=10,
                      color=INK)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
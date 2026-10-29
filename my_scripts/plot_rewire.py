#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_rewire.py

Experiment 10: effect of the fraction p of long-range interactions on the
spatial soup, from the runs of cluster/run_rewire_seeded.sbatch and
cluster/run_rewire_unseeded.sbatch (spatial_survey.py with --rewire-prob p:
every entry of the list of allowed interactions of the grid is replaced, with
probability p, by a random distant program).

Seeded runs (paper_fig4 replicator at the centre of the grid), directories
<results-dir>/rewire_seeded_p<p>/, one per value of p. For each p:
  * survival: fraction of runs in which the seeded replicator reaches 50% of
    the programs (Wilson 95% interval);
  * spread: T50 and T90, the first measured epochs at which 50% and 90% of
    the programs are self-replicators (median and quartiles over the
    surviving runs);
  * spatial structure, from the grid at the end of each surviving run
    (snapshots/seed<N>_final.npz): the mean fraction of identical bytes
    between two adjacent self-replicators minus the same quantity for two
    self-replicators taken at random anywhere in the grid ("neighbour excess
    similarity"); it is 0 when the position on the grid does not matter;
  * diversity: effective number of variants among the self-replicators at
    the end of the run, exp(H) with H the Shannon entropy of the frequencies
    of their distinct 64-byte contents;
  * median fraction of self-replicators over time.

Unseeded runs (random initialization), directories
<results-dir>/rewire_unseeded_p<p>/; the runs of Experiment 8
(<results-dir>/unseeded_240x135, which are the runs with p = 0) and of its
well-mixed control (<results-dir>/wellmixed_32400) are added if present:
  * fraction of runs with a state transition within the run (first crossing
    and end-of-run criteria).

Output: a table on standard output and a figure with one panel per metric.

Usage (from the repository root):
    python3 my_scripts/plot_rewire.py --results-dir spatial_results \
        --out images/rewire.pdf
"""

import argparse
import csv
import glob
import math
import os

import numpy as np

SELFREP_THRESHOLD = 5
INK = "#2b2b2b"
MUTED = "#6b6b6b"
MAIN = "#2a78d6"
REFERENCE = "#8a8a8a"
# One-hue ramp (light to dark) for the curves of the different values of p.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def read_rows(directory):
    path = os.path.join(directory, "summary.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def quartiles(values):
    if len(values) == 0:
        return (float("nan"),) * 3
    return tuple(np.percentile(values, [25, 50, 75]))


def spatial_metrics(path, rng, num_random_pairs=20000):
    """Neighbour excess similarity and effective number of variants of the
    self-replicators in a final snapshot."""
    data = np.load(path)
    soup, selfrep = data["soup"], data["selfrep"] >= SELFREP_THRESHOLD
    rows, cols, _ = soup.shape

    sims = []
    for a, b, ma, mb in (
            (soup[:, :-1], soup[:, 1:], selfrep[:, :-1], selfrep[:, 1:]),
            (soup[:-1, :], soup[1:, :], selfrep[:-1, :], selfrep[1:, :])):
        both = ma & mb
        if both.any():
            sims.append((a[both] == b[both]).mean(axis=1))
    if not sims:
        return float("nan"), float("nan")
    neighbour = float(np.concatenate(sims).mean())

    programs = soup[selfrep]
    if len(programs) < 2:
        return float("nan"), float("nan")
    i = rng.integers(0, len(programs), num_random_pairs)
    j = rng.integers(0, len(programs), num_random_pairs)
    keep = i != j
    random_pairs = float((programs[i[keep]] == programs[j[keep]]).mean())

    _, counts = np.unique(programs, axis=0, return_counts=True)
    freq = counts / counts.sum()
    effective = float(np.exp(-(freq * np.log(freq)).sum()))
    return neighbour - random_pairs, effective


def median_curve(directory, seeds, step):
    """Median fraction of self-replicators over time over the given seeds;
    after a run stops (90% reached) its last value is carried forward."""
    curves = []
    for seed in seeds:
        path = os.path.join(directory, "traces", f"seed{seed}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        epochs = np.array([int(r["epoch"]) for r in rows])
        values = np.array([float(r["selfrep_fraction"] or 0) for r in rows])
        curves.append((epochs, values))
    if not curves:
        return None, None
    end = max(e[-1] for e, _ in curves)
    grid = np.arange(1, end + 1, step)
    stacked = []
    for epochs, values in curves:
        idx = np.searchsorted(epochs, grid, side="right") - 1
        stacked.append(np.where(idx >= 0, values[np.clip(idx, 0, None)], 0.0))
    return grid, np.median(np.array(stacked), axis=0)


def seeded_results(results_dir):
    out = []
    rng = np.random.default_rng(0)
    for d in glob.glob(os.path.join(results_dir, "rewire_seeded_p*")):
        rows = read_rows(d)
        if not rows:
            continue
        p = float(rows[0]["rewire_prob"])
        survived = [r for r in rows if r["epoch_selfrep_50pct"]]
        t50 = np.array([int(r["epoch_selfrep_50pct"]) for r in survived])
        t90 = np.array([int(r["epoch_selfrep_90pct"]) for r in survived
                        if r["epoch_selfrep_90pct"]])
        excess, variants = [], []
        for r in survived:
            path = os.path.join(d, "snapshots", f"seed{r['seed']}_final.npz")
            if os.path.exists(path):
                e, v = spatial_metrics(path, rng)
                if not math.isnan(e):
                    excess.append(e)
                    variants.append(v)
        step = int(rows[0]["callback_interval"])
        grid, curve = median_curve(d, [r["seed"] for r in survived], step)
        out.append({"p": p, "runs": len(rows), "survived": len(survived),
                    "rewired": rewired_fractions(rows),
                    "t50": t50, "t90": t90, "excess": np.array(excess),
                    "variants": np.array(variants), "curve": (grid, curve)})
    return sorted(out, key=lambda c: c["p"])


def rewired_fractions(rows):
    """Fractions of entries of the interaction lists actually replaced, one
    per run (column rewired_fraction of summary.csv)."""
    return np.array([float(r["rewired_fraction"]) for r in rows
                     if r.get("rewired_fraction") not in (None, "")])


def print_rewired(label, p, fractions):
    if p == 0:
        print(f"  {label} p = {p:6.3f}: lists of the grid, nothing replaced")
        return
    if len(fractions) == 0:
        print(f"  {label} p = {p:6.3f}: no rewired_fraction column")
        return
    print(f"  {label} p = {p:6.3f}: {len(fractions)} runs, replaced fraction "
          f"min {fractions.min():.6f}, mean {fractions.mean():.6f}, "
          f"max {fractions.max():.6f}")


def transition_counts(rows):
    n = len(rows)
    first = sum(1 for r in rows if r["transition_epoch"])
    end = sum(1 for r in rows if r["final_entropy"]
              and float(r["final_entropy"]) >= float(r["entropy_threshold"]))
    return n, first, end


def unseeded_results(results_dir):
    out, rewired = {}, {}
    for d in glob.glob(os.path.join(results_dir, "rewire_unseeded_p*")):
        rows = read_rows(d)
        if rows:
            p = float(rows[0]["rewire_prob"])
            out[p] = transition_counts(rows)
            rewired[p] = rewired_fractions(rows)
    rows = read_rows(os.path.join(results_dir, "unseeded_240x135"))
    if rows and 0.0 not in out:
        out[0.0] = transition_counts(rows)
    reference = read_rows(os.path.join(results_dir, "wellmixed_32400"))
    return (sorted(out.items()),
            transition_counts(reference) if reference else None,
            sorted(rewired.items()))


def p_axis(ax):
    ax.set_xscale("symlog", linthresh=1e-3, linscale=0.4)
    ticks = [0, 1e-3, 1e-2, 1e-1, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "0.1%", "1%", "10%", "100%"])
    ax.set_xlim(-2e-4, 1.3)
    ax.set_xlabel("fraction of long-range interactions $p$", color=INK)


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(True, color="#e4e4e4", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def errorbar(ax, x, values, color=MAIN, marker="o", label=None):
    q = np.array([quartiles(v) for v in values])
    ok = ~np.isnan(q[:, 1])
    x = np.asarray(x)[ok]
    q = q[ok]
    ax.errorbar(x, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                fmt=marker + "-", color=color, ms=5, lw=1.5, capsize=3,
                label=label, zorder=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="spatial_results")
    parser.add_argument("--out", default="images/rewire.pdf")
    args = parser.parse_args()

    seeded = seeded_results(args.results_dir)
    unseeded, reference, unseeded_rewired = unseeded_results(args.results_dir)
    if not seeded and not unseeded:
        raise SystemExit(f"no rewire_* results in {args.results_dir}")

    print("Seeded runs (replicator at the centre of the grid)")
    print(f"{'p':>7} {'runs':>5} {'surv.':>6} {'surv. %  [95% CI]':>21} "
          f"{'T50 med [q25,q75]':>19} {'T90 med':>8} {'excess sim.':>12} "
          f"{'variants':>9}")
    for c in seeded:
        lo, hi = wilson(c["survived"], c["runs"])
        q = quartiles(c["t50"])
        t90 = np.median(c["t90"]) if len(c["t90"]) else float("nan")
        ex = np.median(c["excess"]) if len(c["excess"]) else float("nan")
        va = np.median(c["variants"]) if len(c["variants"]) else float("nan")
        print(f"{c['p']:>7.3f} {c['runs']:>5} {c['survived']:>6} "
              f"{100 * c['survived'] / c['runs']:>6.1f}% [{100 * lo:4.1f}, "
              f"{100 * hi:4.1f}] {q[1]:>6.0f} [{q[0]:4.0f},{q[2]:5.0f}] "
              f"{t90:>8.0f} {ex:>12.3f} {va:>9.1f}")
    print("\nSurviving seeded runs that reached 90% of self-replicators")
    for c in seeded:
        print(f"  p = {c['p']:6.3f}: {len(c['t90'])} of {c['survived']}")
    print("\nFraction of entries of the interaction lists actually replaced")
    for c in seeded:
        print_rewired("seeded  ", c["p"], c["rewired"])
    for p, fractions in unseeded_rewired:
        print_rewired("unseeded", p, fractions)
    if unseeded:
        print("\nUnseeded runs (random initialization)")
        for p, (n, first, end) in unseeded:
            lo, hi = wilson(first, n)
            print(f"  p = {p:6.3f}: {first}/{n} first crossing "
                  f"({100 * first / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]), "
                  f"{end}/{n} above the threshold at the end")
        if reference:
            n, first, end = reference
            print(f"  well-mixed reference: {first}/{n} first crossing, "
                  f"{end}/{n} at the end")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    axes = axes.ravel()
    ps = [c["p"] for c in seeded]

    # (a) survival of the seeded replicator
    ax = axes[0]
    if seeded:
        frac = np.array([c["survived"] / c["runs"] for c in seeded])
        ci = np.array([wilson(c["survived"], c["runs"]) for c in seeded])
        ax.errorbar(ps, 100 * frac, yerr=[100 * (frac - ci[:, 0]),
                                         100 * (ci[:, 1] - frac)],
                    fmt="o-", color=MAIN, ms=5, lw=1.5, capsize=3, zorder=3)
    p_axis(ax)
    style(ax, "(a) survival of the seeded replicator", "runs (%)")
    ax.set_ylim(bottom=0)

    # (b) spread time
    ax = axes[1]
    if seeded:
        errorbar(ax, ps, [c["t50"] for c in seeded], label="50% of the programs")
        errorbar(ax, ps, [c["t90"] for c in seeded], color=RAMP[-2],
                 marker="s", label="90% of the programs")
        ax.legend(frameon=False, fontsize=8)
    p_axis(ax)
    style(ax, "(b) time to take over the grid", "epochs")
    ax.set_ylim(bottom=0)

    # (c) neighbour excess similarity
    ax = axes[2]
    if seeded:
        errorbar(ax, ps, [c["excess"] for c in seeded])
    ax.axhline(0, color=REFERENCE, lw=1, ls="--")
    p_axis(ax)
    style(ax, "(c) spatial structure at the end",
          "neighbour excess similarity")

    # (d) effective number of variants
    ax = axes[3]
    if seeded:
        errorbar(ax, ps, [c["variants"] for c in seeded])
    p_axis(ax)
    style(ax, "(d) diversity at the end", "effective number of variants")
    ax.set_ylim(bottom=0)

    # (e) spontaneous transitions
    ax = axes[4]
    if unseeded:
        x = [p for p, _ in unseeded]
        first = np.array([f / n for _, (n, f, _) in unseeded])
        ci = np.array([wilson(f, n) for _, (n, f, _) in unseeded])
        ax.errorbar(x, 100 * first, yerr=[100 * (first - ci[:, 0]),
                                         100 * (ci[:, 1] - first)],
                    fmt="o-", color=MAIN, ms=5, lw=1.5, capsize=3, zorder=3,
                    label="grid")
        if reference:
            n, f, _ = reference
            ax.axhline(100 * f / n, color=REFERENCE, lw=1, ls="--",
                       label="well-mixed, same size")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no unseeded runs yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=9)
    p_axis(ax)
    style(ax, "(e) spontaneous transitions", "runs with a transition (%)")
    ax.set_ylim(bottom=0)

    # (f) fraction of self-replicators over time
    ax = axes[5]
    idx = np.linspace(1, len(RAMP) - 1, max(1, len(seeded))).round().astype(int)
    colors = [RAMP[i] for i in idx]
    for c, color in zip(seeded, colors):
        grid, curve = c["curve"]
        if grid is not None:
            ax.plot(grid - 1, 100 * curve, color=color, lw=1.8,
                    label=f"p = {100 * c['p']:g}%")
    ax.set_xlabel("epoch", color=INK)
    style(ax, "(f) median self-replicators over time", "programs (%)")
    if seeded:
        ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
plot_fig6.py -- Aggregate the mutation-rate-ablation survey CSVs produced by
transition_survey.py and compare them against the paper's Fig. 6 ("fraction
of runs that never reach the state-transition entropy threshold" per
mutation rate). Produces:

  1. A PNG heatmap: paper vs. ours (plus a difference row), one column per
     mutation rate.
  2. LaTeX table rows (rate, n, paper's no-transition %, our no-transition %
     with a Wilson 95% CI, and whether the paper's value falls inside our CI).

This script is standalone: it only needs csv / matplotlib / numpy, no cubff
or CUDA. Run it anywhere the CSVs are available (the cluster, or after
scp-ing the CSVs to your own machine).

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------
Pass one "path:rate" pair per mutation-rate CSV you collected, e.g. (this is
exactly the mapping used throughout the conversation on the cluster):

    python3 plot_fig6.py \
        soup_inspector_results/survey_mut0.csv:0.0 \
        soup_inspector_results/survey_mut012.csv:0.00012 \
        soup_inspector_results/survey_200.csv:0.00024 \
        soup_inspector_results/survey_mut048.csv:0.00048 \
        soup_inspector_results/survey_mut1.csv:0.001 \
        soup_inspector_results/survey_mut5.csv:0.005 \
        soup_inspector_results/survey_mut10.csv:0.01 \
        --out images/fig6_replication.png \
        --tex images/fig6_table_rows.tex

":rate" is optional -- if you omit it the script tries to read the rate
from a "mutation_prob" (or similarly named) column inside the CSV itself.

If your CSV uses column names this script doesn't recognise, it will print
the columns it *did* find and tell you to pass --transition-col / --rate-col
explicitly. It never guesses silently.
---------------------------------------------------------------------------
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# The paper's Fig. 6 numbers (percentage of runs that NEVER reach the
# entropy-based state-transition threshold, at the default 131072-program /
# random-pairing / 16000-epoch setting). Values taken from the paper table
# used throughout this thesis chapter.
# ---------------------------------------------------------------------------
PAPER_RATES = [0.0, 0.00012, 0.00024, 0.00048, 0.001, 0.005, 0.01]
PAPER_LABELS = ["0%", "0.012%", "0.024%\n(default)", "0.048%", "0.1%", "0.5%", "1%"]
PAPER_NO_TRANSITION_PCT = [56.0, 65.0, 60.0, 50.0, 42.0, 62.0, 94.0]
RATE_TOLERANCE = 1e-5

# Column names we'll try, in order, to detect whether a run transitioned.
TRANSITION_EPOCH_COLS = [
    "transition_epoch", "transitioned_epoch", "epoch_transition",
    "first_transition_epoch", "epoch_reached", "epoch_above_threshold",
    "transition_epoch_h0",
]
TRANSITION_BOOL_COLS = [
    "transitioned", "did_transition", "reached_threshold", "state_transition",
]
RATE_COLS = [
    "mutation_prob", "mutation_rate", "mutation_raw", "mut_prob", "p_mutation",
]

EMPTY_TOKENS = {"", "none", "na", "n/a", "null", "-1", "nan"}
TRUE_TOKENS = {"true", "1", "yes", "y"}
FALSE_TOKENS = {"false", "0", "no", "n"}


def wilson_interval(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion k/n. Returns (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def detect_column(fieldnames, candidates):
    lower_map = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def parse_bool_token(tok):
    t = tok.strip().lower()
    if t in TRUE_TOKENS:
        return True
    if t in FALSE_TOKENS:
        return False
    return None


def load_survey(path, transition_col=None, rate_col=None, rate_hint=None):
    """Read one transition_survey.py CSV. Returns (n, k_transitioned, rate)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no data rows found")

    # --- figure out which column tells us "did this seed transition" ---
    epoch_col = transition_col if transition_col else detect_column(fieldnames, TRANSITION_EPOCH_COLS)
    bool_col = None
    if epoch_col is None:
        bool_col = detect_column(fieldnames, TRANSITION_BOOL_COLS)

    if epoch_col is None and bool_col is None:
        raise ValueError(
            f"{path}: could not auto-detect a transition column.\n"
            f"  Columns found: {fieldnames}\n"
            f"  Pass --transition-col <name> to tell the script which column "
            f"to use (either an epoch column, empty/None when the run never "
            f"transitioned, or a boolean column)."
        )

    n = len(rows)
    k = 0
    for row in rows:
        if epoch_col is not None:
            val = (row.get(epoch_col) or "").strip().lower()
            if val not in EMPTY_TOKENS:
                k += 1
        else:
            val = (row.get(bool_col) or "").strip().lower()
            parsed = parse_bool_token(val)
            if parsed:
                k += 1

    # --- figure out the mutation rate for this file ---
    rate = rate_hint
    if rate is None:
        rcol = rate_col if rate_col else detect_column(fieldnames, RATE_COLS)
        if rcol is not None:
            try:
                rate = float(rows[0][rcol])
            except (TypeError, ValueError):
                rate = None
    if rate is None:
        raise ValueError(
            f"{path}: no mutation rate given and none could be read from the "
            f"CSV. Pass it explicitly as path:rate, or use --rate-col."
        )

    return n, k, rate


def nearest_paper_index(rate):
    best_i, best_d = None, None
    for i, r in enumerate(PAPER_RATES):
        d = abs(r - rate)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    if best_d is not None and best_d <= max(RATE_TOLERANCE, rate * 0.05):
        return best_i
    return None


def draw_heatmap(results, out_path):
    """results: list of dicts with keys rate, label, n, k, paper_pct, our_pct, lo, hi"""
    labels = [r["label"] for r in results]
    paper_row = [r["paper_pct"] for r in results]
    our_row = [r["our_pct"] for r in results]
    diff_row = [o - p for o, p in zip(our_row, paper_row)]

    grid = np.array([paper_row, our_row])
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, len(labels) * 1.3), 5),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    im = ax1.imshow(grid, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Paper", "Ours"])
    ax1.set_title("Fraction of runs with NO state transition, by mutation rate")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax1.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                      color="black" if v < 70 else "white")
    fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)

    colors = ["#c0392b" if abs(d) > 10 else "#2c7fb8" for d in diff_row]
    ax2.bar(range(len(labels)), diff_row, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Ours - Paper (pp)")
    ax2.set_title("Difference (positive = we see more non-transitioning runs)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_latex_rows(results):
    lines = []
    inside_count = 0
    for r in results:
        inside = r["lo"] * 100 <= r["paper_pct"] <= r["hi"] * 100
        inside_count += int(inside)
        flag = "" if inside else "  % OUTSIDE our 95% CI"
        lines.append(
            f"{r['rate_str']} & {r['n']} & {r['paper_pct']:.0f}\\% & "
            f"{r['our_pct']:.1f}\\% & "
            f"[{r['lo']*100:.1f}\\%, {r['hi']*100:.1f}\\%] \\\\{flag}"
        )
    summary = f"% Paper's value falls inside our 95% CI for {inside_count}/{len(results)} mutation rates."
    return lines, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="One or more path[:rate] entries")
    ap.add_argument("--out", default="fig6_replication.png", help="Output PNG heatmap path")
    ap.add_argument("--tex", default=None, help="Also write LaTeX table rows to this file")
    ap.add_argument("--transition-col", default=None, help="Force this column name for transition detection")
    ap.add_argument("--rate-col", default=None, help="Force this column name for the mutation rate")
    args = ap.parse_args()

    results = []
    for entry in args.csvs:
        if ":" in entry and not os.path.exists(entry):
            path, rate_str = entry.rsplit(":", 1)
            rate_hint = float(rate_str)
        else:
            path, rate_hint = entry, None

        n, k, rate = load_survey(
            path,
            transition_col=args.transition_col,
            rate_col=args.rate_col,
            rate_hint=rate_hint,
        )
        idx = nearest_paper_index(rate)
        if idx is None:
            print(f"WARNING: {path} has rate {rate} which doesn't match any "
                  f"paper Fig.6 rate within tolerance -- skipping.", file=sys.stderr)
            continue

        no_transition_k = n - k
        our_pct = 100.0 * no_transition_k / n
        lo, hi = wilson_interval(no_transition_k, n)

        rate_pct = rate * 100
        rate_str = f"{rate_pct:g}\\%" if rate_pct != 0 else "0\\%"

        results.append({
            "rate": rate,
            "rate_str": rate_str,
            "label": PAPER_LABELS[idx],
            "n": n,
            "k": k,
            "paper_pct": PAPER_NO_TRANSITION_PCT[idx],
            "our_pct": our_pct,
            "lo": lo,
            "hi": hi,
            "paper_idx": idx,
        })

    if not results:
        print("No usable CSVs -- nothing to plot.", file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda r: r["paper_idx"])

    print(f"{'rate':>14} {'n':>5} {'k(trans.)':>10} {'paper no-trans%':>16} "
          f"{'our no-trans%':>14} {'95% CI':>18}")
    for r in results:
        print(f"{r['rate']*100:>13.4g}% {r['n']:>5} {r['k']:>10} "
              f"{r['paper_pct']:>15.1f}% {r['our_pct']:>13.1f}% "
              f"[{r['lo']*100:>5.1f}%, {r['hi']*100:>5.1f}%]")

    draw_heatmap(results, args.out)
    print(f"\nHeatmap written to {args.out}")

    tex_lines, tex_summary = build_latex_rows(results)
    print("\n--- LaTeX table rows ---")
    for line in tex_lines:
        print(line)
    print(tex_summary)

    if args.tex:
        os.makedirs(os.path.dirname(args.tex) or ".", exist_ok=True)
        with open(args.tex, "w") as f:
            f.write("\n".join(tex_lines) + "\n" + tex_summary + "\n")
        print(f"\nLaTeX rows also written to {args.tex}")


if __name__ == "__main__":
    main()
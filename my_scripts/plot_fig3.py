#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig3.py

Reproduces Fig. 3 of the paper: the interaction between two programs in
which the first self-replicator emerges, replayed step by step.

Input: the file written by find_first_replicator.py, which contains the
128-byte tape of the pair (tape A + tape B) as the engine executes it at the
epoch of emergence, and the tapes after the interaction.

The pair is executed with the bff_noheads interpreter of plot_fig4.py and
every copy instruction ('.' or ',') is recorded. The execution is split into
phases, a phase being a long run of copies in the same direction (from tape
A to tape B, inside tape B, ...). The figure shows the tape at the start and
at the end of every phase; a line connects each byte copied during a phase
to the position it was copied from, as in the paper's figure.

Usage:
    python3 my_scripts/plot_fig3.py \
        --pair soup_inspector_results/tokens/seed171_first_replicator.npz \
        --out soup_inspector_results/fig3_ours.pdf
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import plot_fig4

TAPE = plot_fig4.TAPE_SIZE
PAIR = plot_fig4.PAIR_SIZE

PHASE_COLORS = [(1.00, 0.62, 0.62),   # bytes written in the 1st phase
                (0.66, 0.66, 1.00),   # 2nd phase
                (1.00, 0.85, 0.55),   # 3rd phase
                (0.80, 0.70, 0.90)]   # later phases
LOOP_COLOR = (0.62, 0.88, 0.62)       # instructions executed next


def side(position):
    return "A" if position < TAPE else "B"


def final_tape(states):
    """Tape after the last executed step (states hold the tape before each
    step)."""
    tape, pc, head0, head1 = states[-1]
    tape = bytearray(tape)
    cmd = chr(tape[pc])
    if cmd == "+":
        tape[head0] = (tape[head0] + 1) & 0xFF
    elif cmd == "-":
        tape[head0] = (tape[head0] - 1) & 0xFF
    elif cmd == ".":
        tape[head1] = tape[head0]
    elif cmd == ",":
        tape[head0] = tape[head1]
    return bytes(tape)


def copy_events(states):
    """(step, source, destination) for every executed '.' or ','."""
    events = []
    for k, (tape, pc, head0, head1) in enumerate(states):
        cmd = chr(tape[pc])
        if cmd == ".":
            events.append((k + 1, head0, head1))
        elif cmd == ",":
            events.append((k + 1, head1, head0))
    return events


def find_phases(events, min_copies):
    """Group consecutive copies with the same direction; keep the runs with
    at least `min_copies` copies. Returns a list of dicts."""
    runs = []
    for step, src, dst in events:
        key = (side(src), side(dst))
        if runs and runs[-1]["direction"] == key:
            runs[-1]["last"] = step
            runs[-1]["copies"] += 1
        else:
            runs.append({"direction": key, "first": step, "last": step,
                         "copies": 1})
    return [r for r in runs if r["copies"] >= min_copies]


def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step emergence of the first self-replicator "
                    "(paper Fig. 3).")
    parser.add_argument("--pair", required=True,
                        help="File written by find_first_replicator.py.")
    parser.add_argument("--min-copies", type=int, default=5,
                        help="Minimum number of copies for a run of copies "
                             "to count as a phase.")
    parser.add_argument("--out", default="fig3_ours.pdf")
    args = parser.parse_args()

    data = np.load(args.pair)
    tape0 = data["tape"].tobytes()
    states = plot_fig4.trace_execution(tape0, 8192)
    end = final_tape(states)
    matches = end == data["after"].tobytes()
    print(f"steps executed: {len(states)}")
    print(f"replayed tapes identical to the simulation's: {matches}")
    if not matches:
        sys.exit("the replay differs from the saved result")

    events = copy_events(states)
    phases = find_phases(events, args.min_copies)
    print(f"copy instructions executed: {len(events)}")
    for i, p in enumerate(phases, 1):
        print(f"phase {i}: {p['copies']} copies from tape "
              f"{p['direction'][0]} to tape {p['direction'][1]}, steps "
              f"{p['first']}-{p['last']}")

    # Snapshots: start, end of every phase but the last, end of execution.
    cuts = [0] + [p["last"] for p in phases[:-1]] + [len(states)]
    tapes = [states[c][0] if c < len(states) else end for c in cuts]

    # For every interval between two snapshots: where each byte comes from
    # (position in the previous snapshot), which cells were executed, and in
    # which phase each byte was last written.
    written_in = [None] * PAIR   # phase index of the last write to the cell
    rows = []                    # per snapshot: colours and incoming lines
    rows.append({"written": list(written_in), "lines": []})
    for a, b in zip(cuts[:-1], cuts[1:]):
        source = list(range(PAIR))
        for step, src, dst in events:
            if a < step <= b:
                source[dst] = source[src]
                phase = next((i for i, p in enumerate(phases)
                              if p["first"] <= step <= p["last"]), None)
                written_in[dst] = phase  # None outside the phases
        lines = [(source[i], i) for i in range(PAIR) if source[i] != i]
        rows.append({"written": list(written_in), "lines": lines})
    executed_next = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        # Cells holding an instruction at the start of the interval that
        # are executed during it.
        start_tape = states[a][0]
        cells = {states[k][1] for k in range(a, min(b, len(states)))
                 if start_tape[states[k][1]] in plot_fig4.COMMANDS}
        executed_next.append(cells)
    executed_next.append(set())

    # ---- drawing -----------------------------------------------------------
    cell_w, cell_h, gap, row_gap = 1.0, 2.2, 3.0, 16.0
    label_w = 12.0

    def x_of(position):
        return position * cell_w + (gap if position >= TAPE else 0.0)

    width_in = 10.0
    total_w = label_w + PAIR * cell_w + gap + 1.0
    unit = width_in / total_w
    total_h = len(cuts) * cell_h + (len(cuts) - 1) * row_gap + 16.0
    fig, ax = plt.subplots(figsize=(width_in, total_h * unit))
    ax.set_xlim(-label_w, PAIR * cell_w + gap + 1.0)
    ax.set_ylim(-(total_h - 4.0), 4.0)
    ax.axis("off")
    font = {"family": "DejaVu Sans Mono", "size": 6.5}
    ax.text(x_of(0), 2.2, "Tape A", ha="left", va="center", fontsize=8,
            color="0.35")
    ax.text(x_of(PAIR - 1) + cell_w, 2.2, "Tape B", ha="right", va="center",
            fontsize=8, color="0.35")

    y_top = []
    for r in range(len(cuts)):
        y_top.append(-r * (cell_h + row_gap))
    for r, (cut, tape) in enumerate(zip(cuts, tapes)):
        y = y_top[r]
        ax.text(-1.5, y - cell_h / 2, f"step {cut}", ha="right",
                va="center", fontsize=8)
        for pos in range(PAIR):
            phase = rows[r]["written"][pos]
            if phase is not None:
                color = PHASE_COLORS[min(phase, len(PHASE_COLORS) - 1)]
            elif pos in executed_next[r]:
                color = LOOP_COLOR
            else:
                color = "0.93"
            ax.add_patch(Rectangle((x_of(pos), y - cell_h), cell_w * 0.92,
                                   cell_h, facecolor=color,
                                   edgecolor="none"))
            ch = plot_fig4.cell_text(tape[pos])
            if ch != " ":
                ax.text(x_of(pos) + cell_w * 0.46, y - cell_h / 2, ch,
                        ha="center", va="center", fontdict=font,
                        color="0.4" if ch == "0" else "black")
        if r > 0:
            for src, dst in rows[r]["lines"]:
                ax.plot([x_of(src) + cell_w * 0.46,
                         x_of(dst) + cell_w * 0.46],
                        [y_top[r - 1] - cell_h, y], color="0.45",
                        linewidth=0.35, alpha=0.8)

    handles = [Patch(color=LOOP_COLOR,
                     label="instructions executed until the next row")]
    for i, p in enumerate(phases[:len(PHASE_COLORS)]):
        handles.append(Patch(
            color=PHASE_COLORS[i],
            label=f"copied {p['direction'][0]}$\\rightarrow$"
                  f"{p['direction'][1]} (steps {p['first']}-{p['last']})"))
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
              frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    if png != args.out:
        fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Figure written to {args.out}"
          + (f" and {png}" if png != args.out else ""))
    for cut, tape in zip(cuts, tapes):
        print(f"step {cut:5d}: |{plot_fig4.row_text(tape[:TAPE])}|"
              f"{plot_fig4.row_text(tape[TAPE:])}|")


if __name__ == "__main__":
    main()
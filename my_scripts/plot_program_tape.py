#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_program_tape.py

Two example figures for the terminology of Chapter 4, drawn from the
programs saved by soup_snapshot.py (run of seed 171):

  1. program and tape: two programs, drawn above the 128-cell tape obtained
     by concatenating them, as in the primordial soup;
  2. long tape: an excerpt of 128 cells of a long tape (Experiments 6 and
     7), read from one of the tape dumps written by long_tape_bff.py and
     shown under a bar that represents the whole tape;
  3. interaction: a self-replicator taken from the soup after the state
     transition (program A) is concatenated with a program of the initial
     soup (program B), the tape is executed, and the result is split back
     into two programs. The same two programs are used in both figures.

The tape is executed with the interpreter of plot_bff_example.py, which has
the same semantics as Evaluate in cubff's bff.inc.h (up to 2^13 steps) and
is the interpreter of plot_fig4.py with the length of the tape as a
parameter; plot_bff_example.py --check-engine compares it with the engine.
The self-replication scores of the four programs are computed by the engine.

Programs are drawn as in the other figures: instructions as themselves on a
blue background, the zero byte as '0' on a grey background, any other byte
as a blank.

Usage:
    python3 my_scripts/plot_program_tape.py \
        --data soup_inspector_results/snapshot/seed171_snapshot.npz \
        --out-tape images/program_tape.pdf \
        --out-longtape images/longtape_excerpt.pdf \
        --out-interaction images/interaction_example.pdf
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import plot_bff_example as pbe  # noqa: E402
import token_soup as ts         # noqa: E402

COMMANDS = pbe.COMMANDS
TAPE_SIZE = ts.TAPE_SIZE
MAX_STEPS = ts.MAX_STEPS
INSTR_SHADE = (0.86, 0.90, 0.98)
ZERO_SHADE = (0.93, 0.93, 0.93)


def cell_text(byte):
    if byte == 0:
        return "0"
    if byte in COMMANDS:
        return chr(byte)
    return " "


def row_text(data):
    return "".join(cell_text(b) for b in data)


def program_at(data, epoch, index):
    epochs = list(data["epochs"])
    indices = list(data["program_index"])
    if epoch not in epochs:
        sys.exit(f"epoch {epoch} not in {epochs}")
    if index not in indices:
        sys.exit(f"program {index} not in {indices}")
    return bytes(data["programs"][epochs.index(epoch)][indices.index(index)])


GAP = 3   # cells between two programs drawn side by side


def _cells(ax, x0, y, data, cell_w, cell_h, mono):
    for x, b in enumerate(data):
        shade = (INSTR_SHADE if b in COMMANDS
                 else ZERO_SHADE if b == 0 else None)
        if shade is not None:
            ax.add_patch(Rectangle(((x0 + x) * cell_w, y - cell_h), cell_w,
                                   cell_h, facecolor=shade,
                                   edgecolor="none"))
        t = cell_text(b)
        if t != " ":
            ax.text((x0 + x + 0.5) * cell_w, y - cell_h / 2, t, ha="center",
                    va="center", color="0.45" if b == 0 else "black", **mono)
    ax.add_patch(Rectangle((x0 * cell_w, y - cell_h), len(data) * cell_w,
                           cell_h, facecolor="none", edgecolor="0.6",
                           linewidth=0.4))


def draw_rows(rows, out):
    """rows: list of (label, kind, data, note, header).
    kind 'title': a panel title (data unused, note = text);
    kind 'pair': two 64-byte programs side by side, each above its half of
        the tape (data = (first, second), header = (title1, title2) or None);
    kind 'tape': a 128-cell tape with a dotted line between the two halves;
    kind 'excerpt': a run of cells of a long tape (no boundary line);
    kind 'overview': a bar standing for a whole long tape, with the excerpt
        drawn in the next row marked on it (data = (start, length, size))."""
    total = 2 * TAPE_SIZE + GAP
    cell_w, cell_h = 1.0, 2.6
    label_w, note_w = 17.0, 15.0
    width_in = 10.0
    unit = width_in / (label_w + total * cell_w + note_w)

    def height(r):
        if r[1] == "title":
            return 1.4
        if r[1] == "pair" and r[4]:
            return 2.6
        if r[1] == "overview":
            return 2.4
        return 1.6

    h_total = sum(height(r) for r in rows) + 0.4
    fig, ax = plt.subplots(figsize=(width_in, h_total * cell_h * unit))
    ax.set_xlim(-label_w, total * cell_w + note_w)
    ax.set_ylim(-h_total * cell_h, 0.2 * cell_h)
    ax.axis("off")
    mono = {"family": "DejaVu Sans Mono", "size": 7.0}
    x_tape = GAP / 2           # left edge of a 128-cell tape
    y = 0.0
    for row in rows:
        label, kind, data, note, header = row
        h = height(row)
        if kind == "title":
            ax.text(-label_w, y - 0.6 * cell_h, note, ha="left",
                    va="center", fontsize=8.5, fontweight="bold")
            y -= h * cell_h
            continue
        if kind == "pair":
            if header:
                for k, text in enumerate(header):
                    x0 = k * (TAPE_SIZE + GAP)
                    ax.text((x0 + TAPE_SIZE / 2) * cell_w, y - 0.5 * cell_h,
                            text, ha="center", va="center", fontsize=7.5,
                            color="0.25")
                y -= 1.0 * cell_h
                h -= 1.0
            for k, prog in enumerate(data):
                _cells(ax, k * (TAPE_SIZE + GAP), y, prog, cell_w, cell_h,
                       mono)
        elif kind == "overview":
            start, length, size = data
            width = 2 * TAPE_SIZE
            ax.add_patch(Rectangle((x_tape * cell_w, y - 0.7 * cell_h),
                                   width * cell_w, 0.7 * cell_h,
                                   facecolor="0.92", edgecolor="0.6",
                                   linewidth=0.4))
            xs = x_tape + width * start / size
            xe = x_tape + width * (start + length) / size
            ax.add_patch(Rectangle((xs * cell_w, y - 0.7 * cell_h),
                                   max(xe - xs, 0.4) * cell_w, 0.7 * cell_h,
                                   facecolor="0.2", edgecolor="none"))
            ax.text(x_tape * cell_w, y + 0.25 * cell_h, "cell 0", ha="left",
                    va="bottom", fontsize=6.5, color="0.4")
            ax.text((x_tape + width) * cell_w, y + 0.25 * cell_h,
                    f"cell {size - 1:,}", ha="right", va="bottom",
                    fontsize=6.5, color="0.4")
            # zoom lines towards the excerpt drawn in the next row
            for x_from, x_to in ((xs, x_tape), (xe, x_tape + width)):
                ax.plot([x_from * cell_w, x_to * cell_w],
                        [y - 0.7 * cell_h, y - h * cell_h],
                        color="0.5", linewidth=0.5)
        else:
            _cells(ax, x_tape, y, data, cell_w, cell_h, mono)
            if kind == "tape":
                ax.plot([(x_tape + TAPE_SIZE) * cell_w] * 2,
                        [y + 0.2 * cell_h, y - 1.2 * cell_h],
                        color="0.3", linewidth=0.7, linestyle=":")
        if note:
            ax.text((total + 0.8) * cell_w,
                    y - (0.35 if kind == "overview" else 0.5) * cell_h,
                    note, ha="left", va="center", fontsize=7.5,
                    color="0.25")
        ax.text(-0.8, y - (0.35 if kind == "overview" else 0.5) * cell_h,
                label, ha="right", va="center", fontsize=8)
        y -= h * cell_h
    fig.savefig(out, bbox_inches="tight")
    print(f"written {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--data", default="soup_inspector_results/snapshot/"
                    "seed171_snapshot.npz")
    ap.add_argument("--seed", type=int, default=171)
    ap.add_argument("--replicator-epoch", type=int, default=4001)
    ap.add_argument("--replicator", type=int, default=0)
    ap.add_argument("--food-epoch", type=int, default=0)
    ap.add_argument("--food", type=int, default=1)
    ap.add_argument("--longtape-dump",
                    default="longtape_results/seed3/tape_dumps_seed3/"
                    "tape_000050002267.bin",
                    help="raw tape dump (.bin) written by long_tape_bff.py")
    ap.add_argument("--longtape-start", type=int, default=21312,
                    help="first cell of the excerpt of the long tape")
    ap.add_argument("--out-tape", default="images/program_tape.pdf")
    ap.add_argument("--out-longtape", default="images/longtape_excerpt.pdf")
    ap.add_argument("--out-interaction",
                    default="images/interaction_example.pdf")
    ap.add_argument("--lang", default="bff_noheads")
    args = ap.parse_args()
    data = np.load(args.data)
    for out in (args.out_tape, args.out_longtape, args.out_interaction):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    rep = program_at(data, args.replicator_epoch, args.replicator)
    food = program_at(data, args.food_epoch, args.food)
    print(f"A: program {args.replicator} at epoch {args.replicator_epoch}: "
          f"|{row_text(rep)}|")
    print(f"B: program {args.food} at epoch {args.food_epoch}: "
          f"|{row_text(food)}|")

    # 1. Program and tape (primordial soup).
    draw_rows([("programs", "pair", (rep, food), None,
                ("program A (64 cells)", "program B (64 cells)")),
               ("tape AB", "tape", rep + food, "128 cells", None)],
              args.out_tape)

    # 2. Excerpt of a long tape.
    long_tape = np.fromfile(args.longtape_dump, dtype=np.uint8)
    length = 2 * TAPE_SIZE
    start = args.longtape_start
    if not 0 <= start <= len(long_tape) - length:
        sys.exit(f"--longtape-start must be between 0 and "
                 f"{len(long_tape) - length}")
    excerpt = bytes(long_tape[start:start + length])
    instructions = os.path.basename(args.longtape_dump).split("_")[-1]
    instructions = int(os.path.splitext(instructions)[0])
    print(f"long tape: {args.longtape_dump}, {len(long_tape)} cells, "
          f"after {instructions:,} executed instructions")
    print(f"cells {start}-{start + length - 1}: |{row_text(excerpt)}|")
    draw_rows([("long tape", "overview", (start, length, len(long_tape)),
                f"{len(long_tape):,} cells", None),
               ("excerpt", "excerpt", excerpt,
                f"cells {start:,}\u2013{start + length - 1:,}", None)],
              args.out_longtape)

    # 3. Interaction.
    states, reason = pbe.trace(rep + food, MAX_STEPS)
    after = states[-1][0]
    steps = len(states) - 1
    print(f"interaction: {steps} steps, terminated by: {reason}")

    cubff = ts.import_cubff()
    language = cubff.GetLanguage(args.lang)

    def score(program):
        return int(language.EvalParsedSelfrep(
            cubff.VectorUint8(bytes(program)), args.replicator_epoch,
            args.seed, False))

    scores = [score(rep), score(food), score(after[:TAPE_SIZE]),
              score(after[TAPE_SIZE:])]
    for name, prog, sc in zip(["A ", "B ", "A'", "B'"],
                              [rep, food, after[:TAPE_SIZE],
                               after[TAPE_SIZE:]], scores):
        print(f"{name} |{row_text(prog)}| score {sc}")

    draw_rows([("before", "pair", (rep, food), None,
                (f"A (score {scores[0]})", f"B (score {scores[1]})")),
               ("tape AB", "tape", rep + food, "before execution", None),
               ("tape AB", "tape", after, f"after {steps:,} steps", None),
               ("after", "pair", (after[:TAPE_SIZE], after[TAPE_SIZE:]),
                None, (f"A' (score {scores[2]})",
                       f"B' (score {scores[3]})"))],
              args.out_interaction)


if __name__ == "__main__":
    main()

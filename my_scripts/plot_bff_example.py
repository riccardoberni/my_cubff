#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_bff_example.py

Step-by-step execution of a short example tape, used in Chapter 4 to
illustrate the BFF machine: instructions and no-ops, the instruction pointer,
the two heads and their wrap-around, a loop and the copy instruction.

The example is a tape of 16 cells, shorter than the 128 cells of the soup so
that every step fits in one figure:

    a b 0 { { { [ . > } ] 0 0 0 0 0

The bytes 'a' and 'b' are no-ops holding data, and the zero after them ends
the data. The three '{' move head1 backwards from cell 0, so that it wraps
around to the end of the tape; the loop '[.>}]' then copies the data, one
byte per iteration, until head0 reaches the zero. Finally the instruction
pointer runs past the end of the tape and the execution terminates.

The interpreter is that of plot_fig4.py (Evaluate in cubff's bff.inc.h, for
bff_noheads) with the length of the tape as a parameter. With --check-engine,
the same program followed by zeros up to 128 cells is executed both with this
interpreter and with the engine (bin/main --run), and the tapes are compared
after every step.

Each row of the figure is the state of the tape just BEFORE the given step,
with the instruction pointer (green), head0 (blue) and head1 (red); the
column on the right describes what the step does.

Usage:
    python3 my_scripts/plot_bff_example.py --out images/bff_example.pdf
    python3 my_scripts/plot_bff_example.py --check-engine
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import plot_fig4  # noqa: E402

EXAMPLE = b"ab\x00{{{[.>}]" + bytes(5)
COMMANDS = plot_fig4.COMMANDS
MINUS = "\u2212"


def trace(tape, max_steps=8192):
    """Run `tape` (any length) and return (states, reason).

    states[k] = (tape, pc, head0, head1) just before step k + 1; the last
    element is the state after the last step. Same semantics as
    plot_fig4.trace_execution, which is restricted to 128-cell tapes.
    """
    tape = bytearray(tape)
    size = len(tape)
    pc = head0 = head1 = 0
    states = []
    reason = "step limit"
    for _ in range(max_steps):
        head0 %= size
        head1 %= size
        states.append((bytes(tape), pc, head0, head1))
        cmd = tape[pc]
        if cmd == ord("<"):
            head0 -= 1
        elif cmd == ord(">"):
            head0 += 1
        elif cmd == ord("{"):
            head1 -= 1
        elif cmd == ord("}"):
            head1 += 1
        elif cmd == ord("+"):
            tape[head0] = (tape[head0] + 1) & 0xFF
        elif cmd == ord("-"):
            tape[head0] = (tape[head0] - 1) & 0xFF
        elif cmd == ord("."):
            tape[head1] = tape[head0]
        elif cmd == ord(","):
            tape[head0] = tape[head1]
        elif cmd == ord("["):
            if tape[head0] == 0:
                depth = 1
                pc += 1
                while pc < size and depth > 0:
                    if tape[pc] == ord("]"):
                        depth -= 1
                    if tape[pc] == ord("["):
                        depth += 1
                    pc += 1
                pc -= 1
                if depth != 0:
                    pc = size
        elif cmd == ord("]"):
            if tape[head0] != 0:
                depth = 1
                pc -= 1
                while pc >= 0 and depth > 0:
                    if tape[pc] == ord("]"):
                        depth += 1
                    if tape[pc] == ord("["):
                        depth -= 1
                    pc -= 1
                pc += 1
                if depth != 0:
                    pc = -1
        if pc < 0:
            reason = "unmatched bracket"
            break
        pc += 1
        if pc >= size:
            reason = "end of tape"
            break
    states.append((bytes(tape), pc, head0 % size, head1 % size))
    return states, reason


def show(byte):
    """Commands as themselves, zero as '0', printable no-ops (the data of
    the example) as themselves, any other byte as a blank."""
    if byte == 0:
        return "0"
    if byte in COMMANDS or 0x61 <= byte <= 0x7A:
        return chr(byte)
    return " "


def describe(state, size):
    """What the step starting from `state` does, in a few words."""
    tape, pc, head0, head1 = state
    cmd = tape[pc]
    moves = {ord("<"): ("head0", head0, -1), ord(">"): ("head0", head0, +1),
             ord("{"): ("head1", head1, -1), ord("}"): ("head1", head1, +1)}
    if cmd in moves:
        name, pos, delta = moves[cmd]
        new = (pos + delta) % size
        text = f"{name} {'+' if delta > 0 else MINUS}1"
        if new != pos + delta:
            text += f" (wraps to cell {new})"
        return text
    if cmd == ord("."):
        return (f"copy cell {head0} ('{show(tape[head0])}') "
                f"onto cell {head1}")
    if cmd == ord(","):
        return (f"copy cell {head1} ('{show(tape[head1])}') "
                f"onto cell {head0}")
    if cmd == ord("+"):
        return f"increment cell {head0}"
    if cmd == ord("-"):
        return f"decrement cell {head0}"
    if cmd == ord("["):
        return ("cell under head0 is 0: jump past ']'" if tape[head0] == 0
                else "cell under head0 is not 0: enter the loop")
    if cmd == ord("]"):
        return ("cell under head0 is 0: leave the loop" if tape[head0] == 0
                else "cell under head0 is not 0: jump back to '['")
    return f"no-op '{show(cmd)}': skipped"


def check_engine(main):
    """Compare the interpreter with the engine on the example padded to 128
    cells, after every step."""
    import check_fig4_engine as cfe
    program = EXAMPLE[:11] + bytes(plot_fig4.PAIR_SIZE - 11)
    ours, _ = trace(program)
    fig4 = plot_fig4.trace_execution(program, 8192)
    same_fig4 = [s[0] for s in ours[:-1]] == [s[0] for s in fig4]
    print(f"same states as plot_fig4.trace_execution: {same_fig4} "
          f"({len(fig4)} steps)")
    text = cfe.encode_for_engine(program)
    matches = 0
    for k in range(1, len(ours)):
        tape = cfe.engine_tape_after(main, "bff_noheads", text, k)
        matches += int(tape == ours[k][0])
    print(f"tapes identical to the engine's after each step: "
          f"{matches}/{len(ours) - 1}")


def draw(states, reason, out):
    size = len(states[0][0])
    n_rows = len(states)
    cell_w, cell_h = 1.0, 1.25
    label_w, note_w = 3.0, 18.0
    width_in = 6.0
    unit = width_in / (label_w + size * cell_w + note_w)
    fig, ax = plt.subplots(figsize=(width_in,
                                    (n_rows + 4.5) * cell_h * unit))
    ax.set_xlim(-label_w, size * cell_w + note_w)
    ax.set_ylim(-(n_rows + 3.2) * cell_h, 1.6 * cell_h)
    ax.axis("off")
    mono = {"family": "DejaVu Sans Mono", "size": 7.0}

    ax.text(-1.0, 0.6 * cell_h, "step", ha="right", va="center", fontsize=6.5,
            color="0.3")
    for pos in range(size):
        ax.text((pos + 0.5) * cell_w, 0.6 * cell_h, str(pos), ha="center",
                va="center", fontsize=5.0, color="0.45")
    y = 0.0
    for k, state in enumerate(states):
        tape, pc, head0, head1 = state
        last = k == len(states) - 1
        ax.text(-1.0, y - cell_h / 2, "end" if last else str(k + 1),
                ha="right", va="center", fontsize=6.5)
        marks = {head0: plot_fig4.HEAD0_COLOR}
        marks[head1] = plot_fig4.HEAD1_COLOR
        if 0 <= pc < size:
            marks[pc] = plot_fig4.IP_COLOR
        for pos, color in marks.items():
            ax.add_patch(Rectangle((pos * cell_w, y - cell_h), cell_w,
                                   cell_h, facecolor=color, edgecolor="none"))
        ax.add_patch(Rectangle((0, y - cell_h), size * cell_w, cell_h,
                               facecolor="none", edgecolor="0.8",
                               linewidth=0.3))
        for pos, byte in enumerate(tape):
            ch = show(byte)
            if ch != " ":
                ax.text((pos + 0.5) * cell_w, y - cell_h / 2, ch,
                        ha="center", va="center", fontdict=mono,
                        color="0.45" if ch == "0" else "black")
        note = (f"IP past the end of the tape: execution terminates"
                if last and reason == "end of tape"
                else f"execution terminates ({reason})" if last
                else describe(state, size))
        ax.text(size * cell_w + 0.8, y - cell_h / 2, note, ha="left",
                va="center", fontsize=6.3)
        y -= cell_h

    ax.legend(handles=[Patch(color=plot_fig4.IP_COLOR,
                             label="instruction pointer"),
                       Patch(color=plot_fig4.HEAD0_COLOR, label="head0"),
                       Patch(color=plot_fig4.HEAD1_COLOR, label="head1")],
              loc="lower center", ncol=3, fontsize=6.5, frameon=False,
              bbox_to_anchor=(0.5, -0.01))
    fig.savefig(out, bbox_inches="tight")
    print(f"written {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--out", default="images/bff_example.pdf")
    ap.add_argument("--check-engine", action="store_true",
                    help="compare with the engine (needs bin/main)")
    ap.add_argument("--main", default="bin/main")
    args = ap.parse_args()

    states, reason = trace(EXAMPLE)
    print(f"steps: {len(states) - 1}, terminated by: {reason}")
    for k, s in enumerate(states):
        print(f"{'end' if k == len(states) - 1 else k + 1:>4} "
              f"|{''.join(show(b) for b in s[0])}| ip {s[1]:2d} "
              f"head0 {s[2]:2d} head1 {s[3]:2d}")
    if args.check_engine:
        check_engine(args.main)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    draw(states, reason, args.out)


if __name__ == "__main__":
    main()

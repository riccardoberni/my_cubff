#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig4.py

Reproduces Fig. 4 of the paper: the step-by-step execution of a
self-replicator concatenated with a tape of zeros.

The program is the hand-written self-replicator shown in the paper's Fig. 4
(a 64-byte palindrome: the copy loop, 48 non-coding bytes, the mirrored
loop). It is placed in the first half of a 128-byte tape whose second half
is all zeros, and executed with the semantics of cubff's `bff_noheads`
language (Evaluate in bff.inc.h): both heads and the instruction pointer
start at position 0.

Each row of the figure is the state of the tape just BEFORE the given step
is executed, so row 1 is the initial state, as in the debug output of
`bin/main --run ... --debug` and in the paper's figure.
Coloured cells mark the instruction pointer (green), the read head `head0`
(blue) and the write head `head1` (red).

Usage:
    python3 plot_fig4.py --out fig4_ours.pdf

Besides the figure, the script prints the number of instructions after which
the second half of the tape is an exact copy of the replicator, and compares
our tape with the rows printed in the paper's figure.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

TAPE_SIZE = 64            # bytes per program (kSingleTapeSize in cubff)
PAIR_SIZE = 2 * TAPE_SIZE
COMMANDS = b"[]+-.,<>{}"

# Hand-written self-replicator of the paper's Fig. 4.
PAPER_FIG4_REPLICATOR = b"[[{.>]-]" + b" " * 48 + b"]-]>.{[["

# Rows printed in the paper's Fig. 4, transcribed from the paper: for each
# step, the non-blank characters of the 128-byte tape (the number of blanks
# cannot be recovered reliably from the PDF, so blanks are left out).
_REPLICATOR_TEXT = "[[{.>]-]]-]>.{[["
PAPER_FIG4_ROWS = {}
for _first, _last, _second_tape in [
        (1, 4, "0" * 64),
        (5, 8, "0" * 63 + "["),
        (9, 12, "0" * 62 + "[["),
        (13, 15, "0" * 61 + "{[["),
        (255, 256, "0" + "[{.>]-]]-]>.{[["),
        (257, 258, _REPLICATOR_TEXT),
        (331, 334, _REPLICATOR_TEXT)]:
    for _step in range(_first, _last + 1):
        PAPER_FIG4_ROWS[_step] = _REPLICATOR_TEXT + _second_tape

IP_COLOR = (0.20, 0.80, 0.20)     # instruction pointer
HEAD0_COLOR = (0.55, 0.55, 1.00)  # read head
HEAD1_COLOR = (1.00, 0.30, 0.30)  # write head


def trace_execution(tape, max_steps):
    """Run `tape` (128 bytes) and return one state per step.

    states[k] = (tape, pc, head0, head1) just before step k + 1 is executed.
    Same semantics as Evaluate() in cubff's bff.inc.h for `bff_noheads`.
    """
    tape = bytearray(tape)
    pc = head0 = head1 = 0
    states = []
    for _ in range(max_steps):
        head0 &= PAIR_SIZE - 1
        head1 &= PAIR_SIZE - 1
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
                while pc < PAIR_SIZE and depth > 0:
                    if tape[pc] == ord("]"):
                        depth -= 1
                    if tape[pc] == ord("["):
                        depth += 1
                    pc += 1
                pc -= 1
                if depth != 0:
                    pc = PAIR_SIZE
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
            break
        pc += 1
        if pc >= PAIR_SIZE:
            break
    return states


def cell_text(byte):
    """How one byte is shown: commands as themselves, the null byte as '0',
    any other (non-coding) byte as a blank, as in the paper's figure."""
    if byte == 0:
        return "0"
    if byte in COMMANDS:
        return chr(byte)
    return " "


def row_text(tape):
    return "".join(cell_text(b) for b in tape)


def parse_rows(spec):
    """'1-15,255-258' -> list of groups of consecutive steps."""
    groups = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        groups.append(list(range(int(lo), int(hi or lo) + 1)))
    return groups


def first_complete_copy(states, program):
    """Number of executed instructions after which the second half of the
    tape equals `program` (None if it never happens)."""
    for k, (tape, _, _, _) in enumerate(states):
        if tape[TAPE_SIZE:] == program:
            return k  # states[k] is the state after k instructions
    return None


def compare_with_paper(states):
    """Compare our tape with the rows printed in the paper's Fig. 4."""
    same = [step for step, text in sorted(PAPER_FIG4_ROWS.items())
            if step <= len(states)
            and row_text(states[step - 1][0]).replace(" ", "") == text]
    different = sorted(set(PAPER_FIG4_ROWS) - set(same))
    print(f"rows identical to those printed in the paper's Fig. 4 "
          f"(non-blank characters): {len(same)}/{len(PAPER_FIG4_ROWS)}")
    if different:
        print(f"  rows that differ: {different}")


def draw(states, groups, out):
    n_rows = sum(len(g) for g in groups) + (len(groups) - 1)  # + gap rows
    cell_w, cell_h = 1.0, 2.7
    label_w = 7.0
    width_in = 10.0
    unit = width_in / (label_w + PAIR_SIZE * cell_w + 1.0)
    height_in = (n_rows + 4.0) * cell_h * unit
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(-label_w, PAIR_SIZE * cell_w + 1.0)
    ax.set_ylim(-(n_rows + 3.0) * cell_h, 1.8 * cell_h)
    ax.axis("off")

    font = {"family": "DejaVu Sans Mono", "size": 7.0}
    ax.text(TAPE_SIZE / 2, 1.0 * cell_h, "first tape: self-replicator",
            ha="center", va="center", fontsize=7, color="0.3")
    ax.text(TAPE_SIZE * 1.5, 1.0 * cell_h, "second tape: initially all zeros",
            ha="center", va="center", fontsize=7, color="0.3")

    y = 0.0
    for gi, group in enumerate(groups):
        if gi > 0:
            ax.text(PAIR_SIZE / 2, y - cell_h / 2, "...", ha="center",
                    va="center", fontsize=7)
            y -= cell_h
        for step in group:
            tape, pc, head0, head1 = states[step - 1]
            ax.text(-1.5, y - cell_h / 2, str(step), ha="right", va="center",
                    fontsize=7)
            # Same priority as the engine's debug output: the instruction
            # pointer is drawn over the write head, which is drawn over the
            # read head.
            marks = {head0: HEAD0_COLOR}
            marks[head1] = HEAD1_COLOR
            marks[pc] = IP_COLOR
            for pos, color in marks.items():
                ax.add_patch(Rectangle((pos * cell_w, y - cell_h), cell_w,
                                       cell_h, facecolor=color,
                                       edgecolor="none"))
            for pos, byte in enumerate(tape):
                ch = cell_text(byte)
                if ch != " ":
                    ax.text((pos + 0.5) * cell_w, y - cell_h / 2, ch,
                            ha="center", va="center", fontdict=font,
                            color="0.45" if ch == "0" else "black")
            y -= cell_h
    # Separator between the two tapes.
    ax.plot([TAPE_SIZE, TAPE_SIZE], [0.2 * cell_h, y - 0.2 * cell_h],
            color="0.8", linewidth=0.5, linestyle=":")

    ax.legend(handles=[Patch(color=IP_COLOR, label="instruction pointer"),
                       Patch(color=HEAD0_COLOR, label="read head (head0)"),
                       Patch(color=HEAD1_COLOR, label="write head (head1)")],
              loc="lower center", ncol=3, fontsize=7, frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    fig.savefig(out, dpi=300, bbox_inches="tight")
    png = os.path.splitext(out)[0] + ".png"
    if png != out:
        fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Figure written to {out}" + (f" and {png}" if png != out else ""))


def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step execution of a self-replicator against a "
                    "tape of zeros (paper Fig. 4).")
    parser.add_argument("--program-hex", default=None,
                        help="The 64-byte program as 128 hex digits "
                             "(default: the replicator of the paper's "
                             "Fig. 4).")
    parser.add_argument("--rows", default="1-15,255-258,331-334",
                        help="Steps to show, as in the paper's figure.")
    parser.add_argument("--max-steps", type=int, default=8192,
                        help="Step limit (2^13, as in the soup).")
    parser.add_argument("--out", default="fig4_ours.pdf")
    args = parser.parse_args()

    program = (bytes.fromhex(args.program_hex) if args.program_hex
               else PAPER_FIG4_REPLICATOR)
    if len(program) != TAPE_SIZE:
        raise SystemExit(f"the program must be {TAPE_SIZE} bytes")

    states = trace_execution(program + bytes(TAPE_SIZE), args.max_steps)
    groups = parse_rows(args.rows)
    last = max(max(g) for g in groups)
    if last > len(states):
        raise SystemExit(f"the execution stops after {len(states)} steps; "
                         f"cannot show step {last}")

    done = first_complete_copy(states, program)
    final_tape = states[-1][0]
    print(f"program is a palindrome: {program == program[::-1]}")
    print(f"steps executed: {len(states)}")
    print("second tape is an exact copy of the program after "
          f"{done} instructions" if done is not None else
          "the second tape never becomes a copy of the program")
    print(f"first tape unchanged at the end: {final_tape[:TAPE_SIZE] == program}")
    print(f"second tape equal to the program at the end: "
          f"{final_tape[TAPE_SIZE:] == program}")

    if program == PAPER_FIG4_REPLICATOR:
        compare_with_paper(states)

    draw(states, groups, args.out)


if __name__ == "__main__":
    main()
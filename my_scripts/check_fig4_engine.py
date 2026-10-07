#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_fig4_engine.py

Cross-check of plot_fig4.py against the cubff engine itself.

For every step shown in the paper's Fig. 4, the self-replicator concatenated
with a tape of zeros is executed by the engine (`bin/main --run`, single
program mode) for the corresponding number of instructions, and the tape
printed by the engine at the end is compared with the tape computed by the
Python interpreter of plot_fig4.py.

Usage (from the cubff folder, on a node where bin/main can run):
    python3 -u my_scripts/check_fig4_engine.py --main bin/main
"""

import argparse
import os
import re
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import plot_fig4

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;:]*m")


def encode_for_engine(program):
    """Bytes -> the textual form expected by the engine's BFF parser
    (bff.inc.h): '0' is the null byte, the ten commands are written as
    themselves and any other byte b as the Unicode character U+0100 + b."""
    out = []
    for b in program:
        if b == 0:
            out.append("0")
        elif b in plot_fig4.COMMANDS:
            out.append(chr(b))
        else:
            out.append(chr(0x100 + b))
    return "".join(out)


def decode_engine_line(line):
    """One tape printed by the engine -> bytes (None if it is not a tape)."""
    text = ANSI_ESCAPE.sub("", line)
    if len(text) != plot_fig4.PAIR_SIZE:
        return None
    tape = bytearray()
    for ch in text:
        if ch == "0":
            tape.append(0)
        elif ord(ch) < 0x80 and ord(ch) in plot_fig4.COMMANDS:
            tape.append(ord(ch))
        elif 0x100 <= ord(ch) < 0x200:
            tape.append(ord(ch) - 0x100)
        else:
            return None
    return bytes(tape)


def engine_tape_after(main, lang, program_text, n_instructions):
    """Tape printed by the engine after running `n_instructions` steps."""
    # Arguments are passed as bytes so that the result does not depend on
    # the locale of the node.
    cmd = [os.fsencode(main), b"--lang", lang.encode(), b"--run",
           program_text.encode("utf-8"), b"--run_steps",
           str(n_instructions).encode()]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, timeout=300)
    tapes = [t for t in (decode_engine_line(line) for line in
                         proc.stdout.decode("utf-8", "replace").split("\n"))
             if t is not None]
    # The engine prints the tape before and after the execution.
    return tapes[-1] if len(tapes) >= 2 else None


def main():
    parser = argparse.ArgumentParser(
        description="Compare plot_fig4.py with the cubff engine.")
    parser.add_argument("--main", default="bin/main",
                        help="Path of the engine's command-line tool.")
    parser.add_argument("--lang", default="bff_noheads")
    parser.add_argument("--rows", default="1-15,255-258,331-334",
                        help="Steps to check (default: those of Fig. 4).")
    args = parser.parse_args()

    program = plot_fig4.PAPER_FIG4_REPLICATOR
    states = plot_fig4.trace_execution(
        program + bytes(plot_fig4.TAPE_SIZE), 8192)
    program_text = encode_for_engine(program)

    steps = [s for group in plot_fig4.parse_rows(args.rows) for s in group]
    same = 0
    for step in steps:
        # Row `step` of the figure is the tape after step - 1 instructions.
        ours = states[step - 1][0]
        theirs = engine_tape_after(args.main, args.lang, program_text,
                                   step - 1)
        if theirs is None:
            print(f"step {step}: could not read the engine's output")
        elif theirs == ours:
            same += 1
        else:
            print(f"step {step}: DIFFERENT\n"
                  f"  engine: {plot_fig4.row_text(theirs)}\n"
                  f"  ours:   {plot_fig4.row_text(ours)}")
    print(f"tapes identical to the engine's: {same}/{len(steps)}")


if __name__ == "__main__":
    main()
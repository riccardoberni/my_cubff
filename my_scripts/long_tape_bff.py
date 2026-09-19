# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Long-tape BFF simulator.

This reproduces the "Long tape simulations" setup described in Section 2.4
of Aguera y Arcas et al., "Computational Life: How Well-formed,
Self-replicating Programs Emerge from Simple Interaction" (arXiv:2406.19108).

Unlike the main cubff engine (bff.inc.h / common.h), which is hard-wired
around a "primordial soup" of many fixed-size (kSingleTapeSize = 64 byte)
program pairs that interact by concatenation, this script implements the
alternative environment described in the paper: a SINGLE long, contiguous
tape (default: 65536 bytes) where all code and data live together.

At each iteration we:
  1. pick a random starting position on the tape,
  2. set up head0/head1/pc from that position (head1 gets a configurable
     offset - the paper notes that head1 == head0 == pc makes only
     trivial, non-looping replicators take over, while an offset larger
     than about 8 bytes, e.g. 12 or 16, is needed for LOOPING
     self-replicators to appear),
  3. execute BFF instructions, reading/writing directly on the shared long
     tape, until a step limit is hit or an error occurs (unmatched
     bracket, or the instruction pointer leaves the tape),
  4. optionally apply a low background mutation rate to the tape.

Periodically we measure a "high-order entropy" complexity proxy on the
whole tape (Shannon entropy of byte frequencies minus the compressed
size per byte, i.e. an approximation of Kolmogorov complexity - see
Section 2.1 of the paper) to detect the pre-life -> life state
transition, the same way Figure 11 of the paper does for the long-tape
Forth experiment.

This script is intentionally pure Python (no CUDA, no compilation step)
so that it is easy to run on a normal laptop, at the cost of speed: it
will execute far fewer instructions per second than the paper's C++/CUDA
or Go implementations, but the qualitative behaviour (spontaneous rise of
self-replicators from self-modification) should still be observable.

Metrics and plotting
---------------------
This is a deliberately separate simulation engine from cubff's own
"primordial soup" (bin/main, bff.inc.h): the long-tape execution model
has no notion of a population of programs or of discrete "epochs", so it
cannot share the C++/CUDA engine or the Python bindings (bin/cubff).

It DOES, however, emit metric records in the same dict schema as
`run_metrics.metrics_from_state` (higher_entropy, h0, top1_freq,
top16_freq, frac_selfrep - the last one always NaN here, since "fraction
of the population that is a self-replicator" has no equivalent on a
single shared tape). This means the existing `run_metrics.py` utilities
(`save_records_to_jsonl`, `load_records_from_jsonl`, `plot_run`,
`detect_transition_epoch`) can be reused as-is on long-tape runs too,
using `x_key="instructions"` instead of the default `"epoch"` - so both
engines can be logged, saved and plotted the same way without merging
their (very different) simulation code.

Usage example:
    python3 long_tape_bff.py --tape-size 65536 --head1-offset 16 \
        --max-instructions 20000000 --report-every 200000 \
        --jsonl-out run.jsonl --plot-png run.png
"""

from __future__ import annotations

import argparse
import math
import random
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from paper_constants import PAPER_TRANSITION_ENTROPY_THRESHOLD
from run_metrics import (
    save_records_to_jsonl,
    detect_transition_epoch,
    plot_run,
)

# Try to use brotli (the compressor actually used in the paper / in the
# cubff C++ code, at quality level 2) if it is available; otherwise fall
# back to zlib, which is always present in the Python standard library.
# Both are Lempel-Ziv-style compressors, so either is a valid proxy for
# Kolmogorov complexity in the sense used by the paper (Section 2.1).
try:
    import brotli  # type: ignore

    def compress(data: bytes) -> bytes:
        return brotli.compress(data, quality=2)

    COMPRESSOR_NAME = "brotli (quality=2)"
except ImportError:  # pragma: no cover - environment dependent
    def compress(data: bytes) -> bytes:
        # zlib level 6 is a reasonable, fast, general-purpose default.
        return zlib.compress(bytes(data), level=6)

    COMPRESSOR_NAME = "zlib (level=6, brotli not available)"


# ---------------------------------------------------------------------------
# BFF opcode semantics (mirrors python/bff_interpreter.py in this repo, but
# generalized to operate on an arbitrarily long tape instead of a fixed
# 2 * 64 byte window).
# ---------------------------------------------------------------------------


class BffOp(IntEnum):
    LOOP_START = 0
    LOOP_END = 1
    PLUS = 2
    MINUS = 3
    COPY01 = 4
    COPY10 = 5
    DEC0 = 6
    INC0 = 7
    DEC1 = 8
    INC1 = 9
    NULL = 10
    NOOP = 11


_CHAR_TO_OP = {
    ord('['): BffOp.LOOP_START,
    ord(']'): BffOp.LOOP_END,
    ord('+'): BffOp.PLUS,
    ord('-'): BffOp.MINUS,
    ord('.'): BffOp.COPY01,
    ord(','): BffOp.COPY10,
    ord('<'): BffOp.DEC0,
    ord('>'): BffOp.INC0,
    ord('{'): BffOp.DEC1,
    ord('}'): BffOp.INC1,
    0: BffOp.NULL,
}

# Precompute a 256-entry lookup table for speed (this is the hot path).
_OP_TABLE = [
    _CHAR_TO_OP.get(byte, BffOp.NOOP) for byte in range(256)
]


def get_op_kind(byte: int) -> BffOp:
    return _OP_TABLE[byte]


@dataclass
class RunResult:
    """Outcome of executing one program window on the long tape."""

    instructions_executed: int
    terminated_by_error: bool


def run_one_window(
    tape: bytearray,
    tape_size: int,
    pc0: int,
    head1_offset: int,
    max_steps: int,
) -> RunResult:
    """Execute BFF instructions starting at position pc0 on the long tape.

    Head and instruction-pointer positions wrap around modulo tape_size,
    so execution stays confined to the single shared tape but is free to
    roam across its entire length (unlike the primordial-soup variant,
    where heads wrap within a much smaller 2*kSingleTapeSize window).

    Execution stops when:
      - max_steps instructions have been executed, or
      - the instruction pointer runs past a bracket it cannot match
        (unbalanced '[' or ']'), which we treat as a terminating error,
        exactly like the fixed-tape BFF interpreter does.
    """
    pc = pc0 % tape_size
    head0 = pc0 % tape_size
    head1 = (pc0 + head1_offset) % tape_size

    executed = 0
    terminated_by_error = False

    for _ in range(max_steps):
        cmd = tape[pc]
        kind = get_op_kind(cmd)

        if kind == BffOp.DEC0:
            head0 = (head0 - 1) % tape_size
        elif kind == BffOp.INC0:
            head0 = (head0 + 1) % tape_size
        elif kind == BffOp.DEC1:
            head1 = (head1 - 1) % tape_size
        elif kind == BffOp.INC1:
            head1 = (head1 + 1) % tape_size
        elif kind == BffOp.PLUS:
            tape[head0] = (tape[head0] + 1) & 0xFF
        elif kind == BffOp.MINUS:
            tape[head0] = (tape[head0] - 1) & 0xFF
        elif kind == BffOp.COPY01:
            tape[head1] = tape[head0]
        elif kind == BffOp.COPY10:
            tape[head0] = tape[head1]
        elif kind == BffOp.LOOP_START:
            if get_op_kind(tape[head0]) == BffOp.NULL:
                # Scan forward for the matching ']'.
                depth = 1
                scan = pc + 1
                found = False
                # Bound the scan to the tape size to avoid an infinite
                # loop; this mirrors "terminate execution" on a failed
                # bracket match in the original interpreter.
                for _ in range(tape_size):
                    c = get_op_kind(tape[scan % tape_size])
                    if c == BffOp.LOOP_END:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    elif c == BffOp.LOOP_START:
                        depth += 1
                    scan += 1
                if not found:
                    terminated_by_error = True
                    break
                pc = scan
        elif kind == BffOp.LOOP_END:
            if get_op_kind(tape[head0]) != BffOp.NULL:
                depth = 1
                scan = pc - 1
                found = False
                for _ in range(tape_size):
                    c = get_op_kind(tape[scan % tape_size])
                    if c == BffOp.LOOP_END:
                        depth += 1
                    elif c == BffOp.LOOP_START:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    scan -= 1
                if not found:
                    terminated_by_error = True
                    break
                pc = scan
        # NOOP and NULL fall through with no side effect other than
        # advancing pc below (matching the original semantics, which
        # only skips counting them towards "real" instructions - not
        # relevant here since we count raw steps for simplicity).

        executed += 1
        pc = (pc + 1) % tape_size

    return RunResult(instructions_executed=executed, terminated_by_error=terminated_by_error)


# ---------------------------------------------------------------------------
# Complexity metric (Section 2.1 of the paper): "high-order entropy" =
# Shannon entropy (per-byte) - compressed size per byte. Random noise
# converges to ~0; a tape full of repeated structure (e.g. copies of a
# self-replicator) has substantial positive value.
#
# The dict returned here deliberately uses the SAME keys as
# run_metrics.metrics_from_state (higher_entropy, h0, brotli_bpb,
# top1_freq, top16_freq, frac_selfrep), so that a long-tape run can be
# logged, saved and plotted with the exact same functions used for
# ordinary primordial-soup runs (see module docstring above), even though
# the two engines are otherwise completely separate. "epoch" is
# deliberately NOT one of the keys here: the natural unit of progress on
# a long tape is instructions executed, not epochs - callers should use
# x_key="instructions" when plotting (see plot_run/detect_transition_epoch
# in run_metrics.py).
# ---------------------------------------------------------------------------


def metrics_from_tape(
    tape: bytearray, instructions: int, windows: int, elapsed_s: float
) -> dict:
    counts = [0] * 256
    for b in tape:
        counts[b] += 1
    n = len(tape)

    h0 = 0.0
    for c in counts:
        if c:
            p = c / n
            h0 -= p * math.log2(p)

    compressed_size = len(compress(bytes(tape)))
    brotli_bpb = compressed_size * 8.0 / n  # name kept for schema parity
    higher_entropy = h0 - brotli_bpb

    # Byte-frequency concentration, analogous to the "unique tokens
    # collapse" signal in Figure 1 of the paper: if a self-replicator
    # takes over part of the tape, a handful of bytes become dominant.
    sorted_counts = sorted(counts, reverse=True)
    top1_freq = sorted_counts[0] / n
    top16_freq = sum(sorted_counts[:16]) / n

    return dict(
        instructions=instructions,
        windows=windows,
        higher_entropy=higher_entropy,
        h0=h0,
        brotli_bpb=brotli_bpb,
        top1_freq=top1_freq,
        top16_freq=top16_freq,
        # No equivalent of "fraction of the population that is a
        # self-replicator" on a single shared tape (there is no
        # population of separate programs to score individually).
        frac_selfrep=float("nan"),
        mops_s=(instructions / elapsed_s / 1e6) if elapsed_s > 0 else float("nan"),
    )


# ---------------------------------------------------------------------------
# Main simulation loop.
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    tape_size: int = 65536
    head1_offset: int = 16
    max_steps_per_window: int = 8192
    mutation_every_n_instructions: Optional[int] = 400_000
    max_instructions: int = 50_000_000
    report_every: int = 200_000
    seed: int = 0
    jsonl_path: Optional[str] = None
    plot_path: Optional[str] = None
    entropy_threshold: float = PAPER_TRANSITION_ENTROPY_THRESHOLD
    seed_replicator: Optional[bytes] = None


def random_tape(tape_size: int, rng: random.Random) -> bytearray:
    return bytearray(rng.randrange(256) for _ in range(tape_size))


def apply_random_mutation(tape: bytearray, rng: random.Random) -> None:
    """Flip one random byte on the tape to a uniformly random new value.

    This mirrors the paper's background mutation model (a rare random
    byte replacement), applied at a fixed rate of instructions executed
    rather than per epoch, since there is no notion of "epoch" in the
    long-tape setting.
    """
    pos = rng.randrange(len(tape))
    tape[pos] = rng.randrange(256)


def run_simulation(config: SimConfig) -> None:
    rng = random.Random(config.seed)
    tape = random_tape(config.tape_size, rng)

    if config.seed_replicator is not None:
        if len(config.seed_replicator) > config.tape_size:
            raise ValueError("seed replicator longer than the tape")
        start = rng.randrange(config.tape_size)
        for i, b in enumerate(config.seed_replicator):
            tape[(start + i) % config.tape_size] = b

    total_instructions = 0
    total_windows = 0
    next_report = 0
    next_mutation = (
        config.mutation_every_n_instructions
        if config.mutation_every_n_instructions
        else None
    )

    log_rows: List[dict] = []
    t_start = time.time()

    print(f"[info] compressor in use: {COMPRESSOR_NAME}")
    print(
        f"[info] tape_size={config.tape_size} head1_offset={config.head1_offset} "
        f"max_steps_per_window={config.max_steps_per_window}"
    )

    while total_instructions < config.max_instructions:
        pc0 = rng.randrange(config.tape_size)
        result = run_one_window(
            tape,
            config.tape_size,
            pc0,
            config.head1_offset,
            config.max_steps_per_window,
        )
        total_instructions += result.instructions_executed
        total_windows += 1

        if next_mutation is not None:
            while total_instructions >= next_mutation:
                apply_random_mutation(tape, rng)
                next_mutation += config.mutation_every_n_instructions

        if total_instructions >= next_report:
            elapsed = time.time() - t_start
            record = metrics_from_tape(tape, total_instructions, total_windows, elapsed)
            print(
                f"[report] instructions={record['instructions']:>12,} "
                f"windows={record['windows']:>10,} "
                f"high_order_entropy={record['higher_entropy']:6.3f} "
                f"elapsed={elapsed:7.1f}s "
                f"mops_s={record['mops_s']:6.2f}"
            )
            log_rows.append(record)
            next_report += config.report_every

    transition_at = detect_transition_epoch(
        log_rows, config.entropy_threshold, x_key="instructions"
    )
    if transition_at is not None:
        print(
            f"[info] state transition detected at {transition_at:,} instructions "
            f"(high_order_entropy > {config.entropy_threshold})"
        )
    else:
        print(
            f"[info] no state transition detected within {total_instructions:,} "
            f"instructions (threshold={config.entropy_threshold}); this is a normal "
            f"outcome, not a bug - see paper_constants.py for the paper's own "
            f"transition rates"
        )

    if config.jsonl_path:
        # Same format used by run_metrics/reseed_and_track for soup runs,
        # so load_records_from_jsonl works unmodified on this file too.
        save_records_to_jsonl(log_rows, config.jsonl_path)
        print(f"[info] log written to {config.jsonl_path}")

    if config.plot_path:
        plot_run(
            log_rows,
            title=(
                f"Long-tape BFF (tape_size={config.tape_size}, "
                f"head1_offset={config.head1_offset})"
            ),
            entropy_threshold=config.entropy_threshold,
            save_path=config.plot_path,
            show=False,
            x_key="instructions",
            x_label="instructions executed",
        )
        print(f"[info] plot written to {config.plot_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tape-size", type=int, default=65536,
                    help="length of the single shared tape, in bytes")
    p.add_argument("--head1-offset", type=int, default=16,
                    help="initial offset of head1 relative to the "
                         "execution start position; the paper notes this "
                         "must be larger than ~8 (e.g. 12 or 16) for "
                         "LOOPING self-replicators to arise")
    p.add_argument("--max-steps-per-window", type=int, default=8192,
                    help="max instructions executed per random start "
                         "position before moving to a new one")
    p.add_argument("--mutation-every-n-instructions", type=int, default=400_000,
                    help="apply one random single-byte mutation every N "
                         "instructions executed; set to 0 to disable "
                         "background mutation entirely")
    p.add_argument("--max-instructions", type=int, default=50_000_000,
                    help="total instruction budget for the whole run")
    p.add_argument("--report-every", type=int, default=200_000,
                    help="how often (in instructions) to measure and "
                         "print the high-order entropy of the tape")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--jsonl-out", type=str, default=None,
                    help="optional path to append the complexity trace as "
                         "JSON-lines records (same schema/format as "
                         "run_metrics.save_records_to_jsonl, so it can be "
                         "read back with load_records_from_jsonl)")
    p.add_argument("--plot-png", type=str, default=None,
                    help="optional path to save a PNG plot of the run "
                         "(built with run_metrics.plot_run, the same "
                         "function used for ordinary soup runs)")
    p.add_argument("--entropy-threshold", type=float,
                    default=PAPER_TRANSITION_ENTROPY_THRESHOLD,
                    help="high_order_entropy threshold used to detect the "
                         "state transition; defaults to the paper's own "
                         "criterion (see paper_constants.py)")
    p.add_argument("--seed-replicator", type=str, default=None,
                    help="path to a raw binary file containing a "
                         "hand-crafted (or previously extracted) "
                         "replicator to place at a random position on the "
                         "tape before the run starts, instead of leaving "
                         "the whole tape pure noise. Mirrors the paper's "
                         "own viability check in Section 2.4: 'their "
                         "viability was verified by running an experiment "
                         "where the long-tape was seeded with a "
                         "hand-crafted replicator'. The file can be any "
                         "length up to --tape-size; a 64-byte file "
                         "extracted by soup_inspector.py works too, "
                         "though it was validated for the fixed-tape BFF "
                         "engine and is not guaranteed to loop correctly "
                         "here.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    mutation_rate = (
        args.mutation_every_n_instructions
        if args.mutation_every_n_instructions > 0
        else None
    )

    seed_replicator = None
    if args.seed_replicator:
        with open(args.seed_replicator, "rb") as f:
            seed_replicator = f.read()
        print(
            f"[info] seeding tape with a {len(seed_replicator)}-byte "
            f"replicator from '{args.seed_replicator}' at a random position"
        )

    config = SimConfig(
        tape_size=args.tape_size,
        head1_offset=args.head1_offset,
        max_steps_per_window=args.max_steps_per_window,
        mutation_every_n_instructions=mutation_rate,
        max_instructions=args.max_instructions,
        report_every=args.report_every,
        seed=args.seed,
        jsonl_path=args.jsonl_out,
        plot_path=args.plot_png,
        entropy_threshold=args.entropy_threshold,
        seed_replicator=seed_replicator,
    )
    run_simulation(config)


if __name__ == "__main__":
    main()  
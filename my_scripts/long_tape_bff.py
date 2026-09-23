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
import os
import pickle
import random
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from paper_constants import PAPER_TRANSITION_ENTROPY_THRESHOLD
from run_metrics import (
    save_records_to_jsonl,
    load_records_from_jsonl,
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


# ---------------------------------------------------------------------------
# Human-readable tape rendering, for manual inspection of tape dumps - the
# long-tape equivalent of the tape transcripts in Figure 2/3 of the paper.
# Since this interpreter's opcodes are literal ASCII bytes (unlike cubff's
# packed encoding, see common.h/bff.inc.h), a "readable view" is just: show
# the known operator characters as themselves, '0' for the true zero byte
# (which doubles as the loop-exit/NULL op, same convention the paper's own
# transcripts and soup_inspector.py use), and '.' for any other (non-op,
# "data") byte.
# ---------------------------------------------------------------------------

_SYMBOL_FOR_BYTE = {ord(ch): ch for ch in "[]+-.,<>{}"}
_SYMBOL_FOR_BYTE[0] = '0'


def render_tape_symbols(tape: bytes) -> str:
    return ''.join(_SYMBOL_FOR_BYTE.get(b, '.') for b in tape)


def dump_tape_snapshot(tape: bytearray, total_instructions: int, dump_dir: str) -> None:
    """Writes the current tape to `dump_dir`, both as raw bytes (for
    reloading/further processing) and as a human-readable symbol string
    (for eyeballing where structure/replicators might be forming, the way
    Figure 2 of the paper inspects a soup tape over time). One pair of
    files per call, named by instruction count so a whole run produces an
    ordered history you can flip through afterwards."""
    os.makedirs(dump_dir, exist_ok=True)
    base = os.path.join(dump_dir, f"tape_{total_instructions:012d}")
    with open(base + ".bin", "wb") as f:
        f.write(bytes(tape))
    with open(base + ".txt", "w", encoding="ascii", errors="replace") as f:
        f.write(render_tape_symbols(tape))
        f.write("\n")


# ---------------------------------------------------------------------------
# Full-state checkpointing (tape + RNG state + counters), so a long run can
# be stopped and resumed later WITHOUT recomputing the instructions already
# executed. This is purely an engineering convenience for this Python
# engine (the paper's own long-tape code, in Go, has no such mechanism
# described) - it does not change the simulation's semantics: resuming
# from a checkpoint continues the exact same deterministic trajectory a
# single uninterrupted run with the same --seed would have followed,
# because the RNG state is saved and restored bit-for-bit via
# random.Random.getstate()/setstate(), not just re-seeded.
# ---------------------------------------------------------------------------

_CHECKPOINT_MAGIC = b"LTBFFCK1"


def save_tape_checkpoint(path: str, tape: bytearray, tape_size: int,
                          total_instructions: int, total_windows: int,
                          rng: random.Random) -> None:
    """Writes tape + counters + full RNG state to `path`, atomically (via a
    temp file + rename) so an interrupted write - e.g. Ctrl+C, or a OneDrive
    sync grabbing the file mid-write - can never leave a half-written,
    unreadable checkpoint behind."""
    rng_state_bytes = pickle.dumps(rng.getstate())
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(_CHECKPOINT_MAGIC)
        f.write(struct.pack("<QQQQ", tape_size, total_instructions,
                             total_windows, len(rng_state_bytes)))
        f.write(bytes(tape))
        f.write(rng_state_bytes)
    os.replace(tmp_path, path)  # atomic on both POSIX and Windows


def load_tape_checkpoint(path: str):
    """Returns (tape, tape_size, total_instructions, total_windows, rng)."""
    with open(path, "rb") as f:
        magic = f.read(len(_CHECKPOINT_MAGIC))
        if magic != _CHECKPOINT_MAGIC:
            raise ValueError(
                f"'{path}' is not a long_tape_bff.py checkpoint file "
                f"(bad header) - did you point --resume-from at a "
                f".jsonl or .png file by mistake?"
            )
        tape_size, total_instructions, total_windows, rng_state_len = \
            struct.unpack("<QQQQ", f.read(32))
        tape = bytearray(f.read(tape_size))
        rng_state = pickle.loads(f.read(rng_state_len))
    rng = random.Random()
    rng.setstate(rng_state)
    return tape, tape_size, total_instructions, total_windows, rng


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
    tape: bytearray, instructions: int, windows: int, elapsed_s: float,
    session_instructions: Optional[int] = None,
) -> dict:
    """`instructions`/`windows` are the ABSOLUTE totals (used as the x-axis
    and for the entropy-threshold check), while `session_instructions`, if
    given, is how many instructions were executed since THIS process
    started (elapsed_s always measures THIS process's wall-clock time).
    They only differ when resuming from a checkpoint; mops_s is computed
    from session_instructions so it reflects actual current throughput
    instead of being inflated by instructions from earlier, already-
    finished sessions."""
    if session_instructions is None:
        session_instructions = instructions
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
        mops_s=(session_instructions / elapsed_s / 1e6) if elapsed_s > 0 else float("nan"),
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
    dump_tape_every: Optional[int] = None
    dump_tape_dir: str = "tape_dumps"
    checkpoint_out: Optional[str] = None
    checkpoint_every: Optional[int] = None
    resume_from: Optional[str] = None


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


def run_simulation(config: SimConfig) -> dict:
    """Returns {"seed": ..., "total_instructions": ..., "transition_at":
    ... or None}, mainly so main()'s --num-seeds loop can build a summary
    across runs; the printed output remains the primary way to follow a
    single run."""
    if config.resume_from:
        tape, loaded_tape_size, total_instructions, total_windows, rng = \
            load_tape_checkpoint(config.resume_from)
        if loaded_tape_size != config.tape_size:
            raise ValueError(
                f"--tape-size ({config.tape_size}) does not match the "
                f"checkpoint's own tape size ({loaded_tape_size}); omit "
                f"--tape-size to use whatever size the checkpoint was "
                f"saved with, it cannot be changed mid-run."
            )
        print(f"[info] resumed from checkpoint '{config.resume_from}': "
              f"{total_instructions:,} instructions, {total_windows:,} "
              f"windows already done")
        if config.seed_replicator is not None:
            print("[warn] --seed-replicator is ignored when resuming: the "
                  "tape already contains whatever state was checkpointed.")
    else:
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

    session_start_instructions = total_instructions

    if total_instructions >= config.max_instructions:
        print(f"[info] checkpoint already has {total_instructions:,} "
              f"instructions >= --max-instructions "
              f"({config.max_instructions:,}); nothing to do. Raise "
              f"--max-instructions to continue further.")
        return dict(seed=config.seed, total_instructions=total_instructions,
                    transition_at=None)

    # Mutation/dump/checkpoint schedules are "next multiple of the
    # interval, counted from absolute instruction 0" - NOT "N instructions
    # after this session/resume started". This keeps their timing
    # identical to what a single uninterrupted run would have produced,
    # regardless of how many times it was stopped and resumed, so dumps/
    # checkpoints from different sessions don't collide or skip a slot.
    # (--report-every uses the same rule when resuming, but since a fresh
    # run's very first report can fire before reaching one full interval -
    # see next_report below - the two are not bit-for-bit identical in
    # general; this only affects where report rows land, not the
    # simulation itself.)
    def _next_multiple(current: int, interval: Optional[int]) -> Optional[int]:
        if not interval:
            return None
        return (current // interval + 1) * interval

    next_report = (
        (0 if not config.resume_from
         else _next_multiple(total_instructions, config.report_every))
        if config.report_every else None
    )
    next_mutation = _next_multiple(total_instructions, config.mutation_every_n_instructions)
    next_dump = _next_multiple(total_instructions, config.dump_tape_every)
    next_checkpoint = _next_multiple(total_instructions, config.checkpoint_every)

    log_rows: List[dict] = []
    t_start = time.time()

    print(f"[info] compressor in use: {COMPRESSOR_NAME}")
    print(
        f"[info] tape_size={config.tape_size} head1_offset={config.head1_offset} "
        f"max_steps_per_window={config.max_steps_per_window}"
    )
    if config.dump_tape_every:
        print(f"[info] dumping tape snapshots every "
              f"{config.dump_tape_every:,} instructions to "
              f"'{config.dump_tape_dir}/'")
    if config.checkpoint_out:
        interval_str = (f"every {config.checkpoint_every:,} instructions"
                         if config.checkpoint_every else "at the end only")
        print(f"[info] saving resumable checkpoints to "
              f"'{config.checkpoint_out}' ({interval_str})")

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

        if next_dump is not None:
            while total_instructions >= next_dump:
                dump_tape_snapshot(tape, total_instructions, config.dump_tape_dir)
                next_dump += config.dump_tape_every

        if next_checkpoint is not None:
            while total_instructions >= next_checkpoint:
                save_tape_checkpoint(config.checkpoint_out, tape, config.tape_size,
                                      total_instructions, total_windows, rng)
                next_checkpoint += config.checkpoint_every

        if next_report is not None and total_instructions >= next_report:
            elapsed = time.time() - t_start
            session_instructions = total_instructions - session_start_instructions
            record = metrics_from_tape(tape, total_instructions, total_windows,
                                        elapsed, session_instructions)
            print(
                f"[report] instructions={record['instructions']:>12,} "
                f"windows={record['windows']:>10,} "
                f"high_order_entropy={record['higher_entropy']:6.3f} "
                f"elapsed={elapsed:7.1f}s "
                f"mops_s={record['mops_s']:6.2f}"
            )
            log_rows.append(record)
            if config.jsonl_path:
                # Written incrementally (not just at the end) so an
                # interrupted run - crash, Ctrl+C, or you stopping it on
                # purpose once a checkpoint is saved - never loses the
                # trace recorded so far. Safe to resume into the SAME
                # --jsonl-out path: save_records_to_jsonl appends, so the
                # file accumulates a continuous, gap-free trace across
                # sessions (as long as you keep passing the same path).
                save_records_to_jsonl([record], config.jsonl_path)
            next_report += config.report_every

    # For the final summary and plot, use the FULL history (this session's
    # plus any earlier ones), not just what this process itself computed:
    # since --jsonl-out is now written incrementally and appended to across
    # resumed sessions, reloading it from disk gives the complete trace.
    # Falls back to just this session's in-memory rows if no --jsonl-out
    # was given (nothing was persisted to reload).
    if config.jsonl_path:
        full_log_rows = load_records_from_jsonl(config.jsonl_path)
    else:
        full_log_rows = log_rows

    transition_at = detect_transition_epoch(
        full_log_rows, config.entropy_threshold, x_key="instructions"
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
        print(f"[info] log up to date at {config.jsonl_path} "
              f"({len(full_log_rows)} records total)")

    if config.checkpoint_out:
        # Always save a final checkpoint at the end too, regardless of
        # --checkpoint-every, so the run can always be continued later
        # even if it happened to finish between two scheduled saves.
        save_tape_checkpoint(config.checkpoint_out, tape, config.tape_size,
                              total_instructions, total_windows, rng)
        print(f"[info] final checkpoint written to {config.checkpoint_out} "
              f"(resume with --resume-from {config.checkpoint_out})")

    if config.plot_path:
        plot_run(
            full_log_rows,
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

    return dict(seed=config.seed, total_instructions=total_instructions,
                transition_at=transition_at)


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
    p.add_argument("--jsonl-out", type=str,
                    default="seed{seed}_{instr}instr.jsonl",
                    help="path to append the complexity trace as "
                         "JSON-lines records (same schema/format as "
                         "run_metrics.save_records_to_jsonl, so it can be "
                         "read back with load_records_from_jsonl). ON by "
                         "default, self-named from THIS run's seed and "
                         "--max-instructions - e.g. seed 5 with "
                         "--max-instructions 2000000000 writes "
                         "'seed5_2000000000instr.jsonl' (still placed "
                         "inside 'seed5/', see --output-dir; the name "
                         "stays self-describing even if the file is later "
                         "copied out of that folder). {seed}/{instr}/"
                         "{instr_m}/{instr_b} placeholders (raw, in "
                         "millions rounded, in billions rounded to 1 "
                         "decimal) are filled in wherever they appear in "
                         "whatever value you pass. Pass --jsonl-out \"\" "
                         "(empty string) to turn this off entirely.")
    p.add_argument("--plot-png", type=str,
                    default="seed{seed}_{instr}instr.png",
                    help="path to save a PNG plot of the run (built with "
                         "run_metrics.plot_run). ON by default, same "
                         "self-naming and {seed}/{instr}/{instr_m}/"
                         "{instr_b} placeholders as --jsonl-out; pass "
                         "--plot-png \"\" to turn off.")
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
                         "the whole tape pure noise. NOTE: the paper's own "
                         "seeded long-tape viability check (Section 3.1.2) "
                         "used a hand-crafted FORTH program on a different "
                         "instruction set, not BFF; this flag is an "
                         "analogous experiment on THIS engine, not a "
                         "reproduction of a published number. The file can "
                         "be any length up to --tape-size; a 64-byte file "
                         "extracted by soup_inspector.py works too, though "
                         "it was bred in a fixed 2x64-byte window with "
                         "heads starting at position 0, so it is not "
                         "guaranteed to loop correctly when dropped at a "
                         "random position with head1 at +head1_offset "
                         "here.")
    p.add_argument("--dump-tape-every", type=int, default=None,
                    help="if set, saves a snapshot of the WHOLE tape every "
                         "N instructions into --dump-tape-dir: one raw "
                         ".bin file (the exact bytes) and one .txt file "
                         "(a human-readable symbol rendering - BFF "
                         "operators shown as themselves, '0' for the true "
                         "zero byte, '.' for any other data byte) per "
                         "snapshot, named by instruction count. This is "
                         "the long-tape equivalent of manually inspecting "
                         "a tape over time the way Figure 2 of the paper "
                         "does for the soup - it does not attempt to "
                         "automatically locate or extract a replicator, "
                         "just gives you the raw material to look at "
                         "yourself. Off by default: for a 65536-byte tape "
                         "this is 65KB+65KB per snapshot, so pick an "
                         "interval that won't fill your disk on a long run.")
    p.add_argument("--dump-tape-dir", type=str,
                    default="tape_dumps_seed{seed}",
                    help="folder for --dump-tape-every snapshots (created "
                         "if missing). Defaults to 'tape_dumps_seed{seed}' "
                         "rather than a plain 'tape_dumps': the individual "
                         "snapshot files inside are named ONLY by "
                         "instruction count (e.g. "
                         "'tape_001785006048.bin'), with no seed in the "
                         "file name itself, so if one is ever copied out "
                         "on its own (to inspect it, to hand it to "
                         "someone) the containing folder's name is the "
                         "only remaining record of which seed it came "
                         "from. Supports the same {seed}/{instr}/"
                         "{instr_m}/{instr_b} placeholders as --jsonl-out.")
    p.add_argument("--checkpoint-out", type=str,
                    default="seed{seed}_{instr}instr_ckpt.dat",
                    help="path to save the FULL simulation state (tape + "
                         "RNG state + counters), so the run can be "
                         "continued later with --resume-from WITHOUT "
                         "recomputing the instructions already executed. "
                         "ON by default, same self-naming and {seed}/"
                         "{instr}/{instr_m}/{instr_b} placeholders as "
                         "--jsonl-out ({instr} is filled in with "
                         "--max-instructions, i.e. the run's TARGET total, "
                         "same value whether this is a fresh run or one "
                         "continued with --resume-from); pass "
                         "--checkpoint-out \"\" to turn off. Always "
                         "written once at the end of the run regardless "
                         "of --checkpoint-every; combine with "
                         "--checkpoint-every to also save periodically "
                         "during a long run (recommended - otherwise you "
                         "only get a checkpoint if the run reaches "
                         "--max-instructions normally, which doesn't help "
                         "if you kill it early). Each save OVERWRITES the "
                         "same file (atomically, via a temp file + "
                         "rename), it does not keep a history - use "
                         "--dump-tape-every for that.")
    p.add_argument("--checkpoint-every", type=int, default=None,
                    help="how often (in instructions) to overwrite "
                         "--checkpoint-out during the run; requires "
                         "--checkpoint-out. Without this, only the final "
                         "checkpoint at the end of the run is saved.")
    p.add_argument("--output-dir", type=str, default="longtape_results",
                    help="base folder for every output this run produces. "
                         "Inside it, EACH SEED GETS ITS OWN SUBFOLDER "
                         "'seed<N>/' holding that seed's --jsonl-out, "
                         "--plot-png, --checkpoint-out and "
                         "--dump-tape-dir together (created automatically "
                         "if missing) - this applies even with the "
                         "default --num-seeds 1, so the layout is always "
                         "'<output-dir>/seed<N>/...', never files loose in "
                         "the current directory. Defaults to "
                         "'longtape_results', so this is ON by default - "
                         "pass --output-dir \"\" (empty string) to disable "
                         "it and write exactly where each flag's own path "
                         "says instead. A relative path given to one of "
                         "those flags is joined onto the seed's subfolder; "
                         "an absolute path is left exactly as given "
                         "(escapes --output-dir on purpose, e.g. to share "
                         "a checkpoint from elsewhere). Does NOT affect "
                         "--resume-from or --seed-replicator - those are "
                         "inputs you point at explicitly, not outputs to "
                         "collect.")
    p.add_argument("--num-seeds", type=int, default=1,
                    help="run this many INDEPENDENT seeds, one full "
                         "simulation after another, starting at --seed and "
                         "incrementing by 1 each time (--seed, --seed+1, "
                         "..., --seed+num_seeds-1). This is for the 'how "
                         "OFTEN does this happen' question - the paper's "
                         "own transition rates (paper_constants.py) are "
                         "always measured over many independent runs, "
                         "never a single one, and a single run here tells "
                         "you nothing about whether what it found is "
                         "typical or a one-off. NOT a substitute for "
                         "--resume-from/checkpoints, which continue ONE "
                         "specific run further rather than sampling new "
                         "ones (incompatible with --resume-from for that "
                         "reason). Output paths (--jsonl-out, --plot-png, "
                         "--checkpoint-out, --dump-tape-dir) automatically "
                         "get '_seedN' inserted so runs don't overwrite "
                         "each other's files, and a summary table + "
                         "transition rate is printed across all seeds at "
                         "the end.")
    p.add_argument("--resume-from", type=str, default=None,
                    help="path to a checkpoint previously written by "
                         "--checkpoint-out; if given, the tape and RNG "
                         "state are loaded from it instead of being freshly "
                         "initialized, and the run CONTINUES the exact "
                         "same deterministic trajectory from there. "
                         "--seed/--seed-replicator/--tape-size are ignored "
                         "(the checkpoint already encodes the tape size and "
                         "everything about the tape's content); "
                         "--max-instructions is still the TOTAL target "
                         "(including instructions already done before the "
                         "checkpoint), not an additional amount to run.")
    return p


def _format_path(path: Optional[str], seed: int, max_instructions: int) -> Optional[str]:
    """Fills in {seed}/{instr}/{instr_m}/{instr_b} placeholders in a
    --jsonl-out/--plot-png/--checkpoint-out/--dump-tape-dir value, so a
    self-descriptive name like 'seed{seed}_{instr}instr.jsonl' becomes
    e.g. 'seed5_2000000000instr.jsonl' for that specific run - useful
    since files sometimes get copied out of their 'seed<N>/' folder (for
    a thesis, an email, ...) and lose the context the folder name gave
    them. A plain name with no placeholders (e.g. 'run.jsonl') is
    returned unchanged, exactly as before this existed."""
    if path is None:
        return None
    return path.format(
        seed=seed,
        instr=max_instructions,
        instr_m=round(max_instructions / 1e6),
        instr_b=round(max_instructions / 1e9, 1),
    )


def _in_dir(path: Optional[str], dir_: Optional[str]) -> Optional[str]:
    """Joins `path` onto `dir_`, unless `path` is already absolute (left
    untouched on purpose - an escape hatch to point somewhere else
    entirely, e.g. a checkpoint shared from another folder) or `dir_`/
    `path` is None."""
    if path is None or not dir_:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(dir_, path)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    mutation_rate = (
        args.mutation_every_n_instructions
        if args.mutation_every_n_instructions > 0
        else None
    )

    if args.checkpoint_every and not args.checkpoint_out:
        sys.exit("Error: --checkpoint-every requires --checkpoint-out.")
    if args.num_seeds > 1 and args.resume_from:
        sys.exit(
            "Error: --num-seeds > 1 and --resume-from are mutually "
            "exclusive - resuming continues ONE specific run further, "
            "it is not a way to sample new independent ones. Use "
            "--num-seeds without --resume-from to sample several fresh "
            "runs, or --resume-from alone (--num-seeds 1, the default) "
            "to continue a specific one."
        )

    seed_replicator = None
    if args.seed_replicator:
        if args.resume_from:
            sys.stderr.write(
                "Note: --seed-replicator is ignored with --resume-from "
                "(see run_simulation's own warning once the checkpoint is "
                "actually loaded).\n"
            )
        else:
            with open(args.seed_replicator, "rb") as f:
                seed_replicator = f.read()
            print(
                f"[info] seeding tape with a {len(seed_replicator)}-byte "
                f"replicator from '{args.seed_replicator}' at a random position"
            )

    results = []
    for i in range(args.num_seeds):
        seed = args.seed + i
        if args.num_seeds > 1:
            print(f"\n=== seed {seed} ({i + 1}/{args.num_seeds}) ===")

        # Everything this seed produces (checkpoint, tape dumps, jsonl,
        # plot) goes in its OWN subfolder under --output-dir, e.g.
        # 'longtape_results/seed3/run.jsonl' - one self-contained folder
        # per run/seed instead of everything mixed together with '_seedN'
        # suffixes in the filenames. Applies even with --num-seeds 1 (a
        # single run still gets its own 'seed<N>' folder), so the layout
        # is always the same shape regardless of how many seeds you ran.
        if args.output_dir:
            seed_dir = os.path.join(args.output_dir, f"seed{seed}")
            os.makedirs(seed_dir, exist_ok=True)
        else:
            seed_dir = None

        # "" (explicit opt-out, e.g. --jsonl-out "") must become None here,
        # BEFORE _in_dir joins it onto seed_dir - otherwise os.path.join
        # would turn "" into just the directory path itself (e.g.
        # 'longtape_results/seed5/'), which is a real path, not "off".
        jsonl_out = _format_path(args.jsonl_out, seed, args.max_instructions) or None
        plot_png = _format_path(args.plot_png, seed, args.max_instructions) or None
        checkpoint_out = _format_path(args.checkpoint_out, seed, args.max_instructions) or None
        dump_tape_dir = _format_path(args.dump_tape_dir, seed, args.max_instructions) or None

        config = SimConfig(
            tape_size=args.tape_size,
            head1_offset=args.head1_offset,
            max_steps_per_window=args.max_steps_per_window,
            mutation_every_n_instructions=mutation_rate,
            max_instructions=args.max_instructions,
            report_every=args.report_every,
            seed=seed,
            jsonl_path=_in_dir(jsonl_out, seed_dir),
            plot_path=_in_dir(plot_png, seed_dir),
            entropy_threshold=args.entropy_threshold,
            seed_replicator=seed_replicator,
            dump_tape_every=args.dump_tape_every,
            dump_tape_dir=_in_dir(dump_tape_dir, seed_dir),
            checkpoint_out=_in_dir(checkpoint_out, seed_dir),
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume_from,
        )
        results.append(run_simulation(config))

    if args.num_seeds > 1:
        print(f"\n=== summary across {args.num_seeds} seeds "
              f"({args.seed}..{args.seed + args.num_seeds - 1}) ===")
        for r in results:
            status = (f"transition at {r['transition_at']:,} instructions"
                       if r["transition_at"] is not None else "no transition")
            print(f"  seed {r['seed']:>4}: {status} "
                  f"({r['total_instructions']:,} instructions total)")
        n_transitions = sum(1 for r in results if r["transition_at"] is not None)
        rate = n_transitions / len(results) if results else float("nan")
        print(f"\nTransition rate: {rate:.1%} ({n_transitions}/{len(results)}) "
              f"within {args.max_instructions:,} instructions each.")
        print("(This is YOUR rate for these parameters/engine, not "
              "directly the paper's own soup numbers in paper_constants.py "
              "- the paper does not report a long-tape BFF transition "
              "rate to compare against; see soup_inspector.py for that "
              "comparison instead.)")


if __name__ == "__main__":
    main()
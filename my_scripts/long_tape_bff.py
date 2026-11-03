"""Long-tape BFF simulator (Experiments 6 and 7).

Long-tape setting of Aguera y Arcas et al., "Computational Life: How
Well-formed, Self-replicating Programs Emerge from Simple Interaction"
(arXiv:2406.19108, Section 2.4). The paper does not release this
environment for BFF as part of cubff, and the engine (bin/main, bff.inc.h)
only handles a soup of 64-byte programs that interact in pairs, so the
environment is implemented here in pure Python.

A single tape (default 65536 bytes) holds all code and data. At each
iteration:
  1. a random start position is chosen on the tape;
  2. the instruction pointer and head0 start there, head1 at a fixed offset
     (--head1-offset, default 16; the paper reports that with no offset only
     trivial non-looping replicators take over, while an offset somewhat
     larger than 8 allows looping ones);
  3. BFF instructions are executed on the tape, with positions wrapping
     around its end, until the step limit of the window is reached or a
     bracket cannot be matched;
  4. every --mutation-every-n-instructions executed steps, one random byte
     of the tape is replaced by a random value.
Steps are counted as positions visited by the instruction pointer, including
the bytes that are not instructions, as in the step limit of the engine.

Every --report-every steps the high-order entropy of the whole tape is
measured (Shannon entropy of the byte frequencies minus compressed bits per
byte) together with the fraction of the tape occupied by the most common
byte and by the 16 most common bytes. The compressed size is computed with
zlib at level 6 (--compressor zlib, the default, used for Experiments 6 and
7) or with brotli at quality 2 as in the engine (--compressor brotli, needs
the Python module brotli).

Outputs, in <output-dir>/seed<N>/: the trace as JSON lines, a figure
(plot_longtape.py), a checkpoint from which the run can be resumed
exactly (tape, counters and state of the random generator) and, with
--dump-tape-every, snapshots of the whole tape.

Usage example:
    python3 -u my_scripts/long_tape_bff.py --seed 0 \
        --max-instructions 2000000000 --report-every 5000000 \
        --dump-tape-every 50000000 --output-dir longtape_results
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

from plot_longtape import (
    THRESHOLD,
    first_crossing,
    load_records,
    plot_longtape,
    save_records,
)

# Compressor used for the high-order entropy, chosen with --compressor.
COMPRESSOR_NAME = "zlib (level=6)"


def compress(data: bytes) -> bytes:
    return zlib.compress(bytes(data), level=6)


def use_brotli() -> None:
    """Compute the compressed size with brotli at quality 2, as the engine."""
    global COMPRESSOR_NAME, compress
    import brotli  # type: ignore

    def brotli_compress(data: bytes) -> bytes:
        return brotli.compress(bytes(data), quality=2)

    compress = brotli_compress
    COMPRESSOR_NAME = "brotli (quality=2)"


# ---------------------------------------------------------------------------
# BFF opcode semantics, as in python/bff_interpreter.py, on a tape of any
# length instead of the 2 * 64 byte window of the soup.
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
# Text rendering of the tape snapshots: commands as themselves, '0' for the
# zero byte and '.' for every other byte.
# ---------------------------------------------------------------------------

_SYMBOL_FOR_BYTE = {ord(ch): ch for ch in "[]+-.,<>{}"}
_SYMBOL_FOR_BYTE[0] = '0'


def render_tape_symbols(tape: bytes) -> str:
    return ''.join(_SYMBOL_FOR_BYTE.get(b, '.') for b in tape)


def dump_tape_snapshot(tape: bytearray, total_instructions: int, dump_dir: str) -> None:
    """Write the tape to `dump_dir` as raw bytes (.bin) and as text (.txt),
    in files named after the number of executed instructions."""
    os.makedirs(dump_dir, exist_ok=True)
    base = os.path.join(dump_dir, f"tape_{total_instructions:012d}")
    with open(base + ".bin", "wb") as f:
        f.write(bytes(tape))
    with open(base + ".txt", "w", encoding="ascii", errors="replace") as f:
        f.write(render_tape_symbols(tape))
        f.write("\n")


# ---------------------------------------------------------------------------
# Checkpoints (tape, counters and state of the random generator): a run
# resumed from a checkpoint follows exactly the trajectory of an
# uninterrupted run with the same seed.
# ---------------------------------------------------------------------------

_CHECKPOINT_MAGIC = b"LTBFFCK1"


def save_tape_checkpoint(path: str, tape: bytearray, tape_size: int,
                          total_instructions: int, total_windows: int,
                          rng: random.Random) -> None:
    """Write tape, counters and random-generator state to `path` through a
    temporary file, so that an interrupted write never leaves a corrupted
    checkpoint."""
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
                f"(bad header)"
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

    Execution stops after max_steps steps or when a bracket cannot be
    matched, as in the engine.
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
        # NOOP and NULL only advance pc. As in the engine's step limit,
        # every step counts, including the bytes that are not instructions.

        executed += 1
        pc = (pc + 1) % tape_size

    return RunResult(instructions_executed=executed, terminated_by_error=terminated_by_error)


# ---------------------------------------------------------------------------
# High-order entropy (Section 2.1 of the paper): Shannon entropy of the
# bytes minus compressed bits per byte, on the whole tape.
# ---------------------------------------------------------------------------


def metrics_from_tape(
    tape: bytearray, instructions: int, windows: int, elapsed_s: float,
    session_instructions: Optional[int] = None,
) -> dict:
    """`instructions` and `windows` are totals since the start of the run;
    `session_instructions` counts only the steps of the current process
    (they differ after a resume) and is used for the speed mops_s."""
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
    brotli_bpb = compressed_size * 8.0 / n  # name kept also with zlib
    higher_entropy = h0 - brotli_bpb

    # Concentration of the byte frequencies.
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
        # No self-replication check on the long tape.
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
    entropy_threshold: float = THRESHOLD
    seed_replicator: Optional[bytes] = None
    dump_tape_every: Optional[int] = None
    dump_tape_dir: str = "tape_dumps"
    checkpoint_out: Optional[str] = None
    checkpoint_every: Optional[int] = None
    resume_from: Optional[str] = None


def random_tape(tape_size: int, rng: random.Random) -> bytearray:
    return bytearray(rng.randrange(256) for _ in range(tape_size))


def apply_random_mutation(tape: bytearray, rng: random.Random) -> None:
    """Replace one random byte of the tape with a random value."""
    pos = rng.randrange(len(tape))
    tape[pos] = rng.randrange(256)


def run_simulation(config: SimConfig) -> dict:
    """Run one seed; return its seed, total instructions and first
    crossing of the threshold (or None)."""
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

    # Mutations, dumps and checkpoints happen at multiples of their interval
    # counted from the start of the run, so that a resumed run follows the
    # same schedule as an uninterrupted one.
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
                # Appended at every measurement, so that the trace of an
                # interrupted or resumed run is complete.
                save_records([record], config.jsonl_path)
            next_report += config.report_every

    # The whole trace, including earlier sessions of a resumed run.
    if config.jsonl_path:
        full_log_rows = load_records(config.jsonl_path)
    else:
        full_log_rows = log_rows

    transition_at = first_crossing(full_log_rows, config.entropy_threshold)
    if transition_at is not None:
        print(
            f"[info] state transition detected at {transition_at:,} instructions "
            f"(high_order_entropy > {config.entropy_threshold})"
        )
    else:
        print(
            f"[info] no state transition detected within {total_instructions:,} "
            f"instructions (threshold={config.entropy_threshold})"
        )

    if config.jsonl_path:
        print(f"[info] log up to date at {config.jsonl_path} "
              f"({len(full_log_rows)} records total)")

    if config.checkpoint_out:
        # Final checkpoint, so that the run can be continued later.
        save_tape_checkpoint(config.checkpoint_out, tape, config.tape_size,
                              total_instructions, total_windows, rng)
        print(f"[info] final checkpoint written to {config.checkpoint_out} "
              f"(resume with --resume-from {config.checkpoint_out})")

    if config.plot_path:
        plot_longtape(full_log_rows, config.plot_path,
                      config.entropy_threshold)
        print(f"[info] plot written to {config.plot_path}")

    return dict(seed=config.seed, total_instructions=total_instructions,
                transition_at=transition_at)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tape-size", type=int, default=65536,
                   help="length of the tape, in bytes")
    p.add_argument("--head1-offset", type=int, default=16,
                   help="initial offset of head1 from the start position")
    p.add_argument("--max-steps-per-window", type=int, default=8192,
                   help="step limit for each random start position")
    p.add_argument("--mutation-every-n-instructions", type=int,
                   default=400_000,
                   help="one random byte replaced every N executed steps "
                        "(0: no mutations)")
    p.add_argument("--max-instructions", type=int, default=50_000_000,
                   help="total number of steps of the run (with "
                        "--resume-from: total including the steps already "
                        "done)")
    p.add_argument("--report-every", type=int, default=200_000,
                   help="interval, in steps, between two measurements")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the random generator")
    p.add_argument("--jsonl-out", type=str,
                   default="seed{seed}_{instr}instr.jsonl",
                   help="trace file (JSON lines, appended); placeholders "
                        "{seed}, {instr}, {instr_m}, {instr_b}; empty "
                        "string to disable")
    p.add_argument("--plot-png", type=str,
                   default="seed{seed}_{instr}instr.png",
                   help="figure of the run (plot_longtape.py); same "
                        "placeholders; empty string to disable")
    p.add_argument("--entropy-threshold", type=float, default=THRESHOLD,
                   help="threshold on the high-order entropy (default 1, "
                        "as in the paper)")
    p.add_argument("--seed-replicator", type=str, default=None,
                   help="binary file with a program to write at a random "
                        "position of the initial tape")
    p.add_argument("--dump-tape-every", type=int, default=None,
                   help="save the whole tape (.bin and .txt) every N steps")
    p.add_argument("--dump-tape-dir", type=str,
                   default="tape_dumps_seed{seed}",
                   help="folder of the tape snapshots; same placeholders")
    p.add_argument("--checkpoint-out", type=str,
                   default="seed{seed}_{instr}instr_ckpt.dat",
                   help="checkpoint file, overwritten at every save and "
                        "written at the end of the run; same placeholders; "
                        "empty string to disable")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="also save the checkpoint every N steps")
    p.add_argument("--output-dir", type=str, default="longtape_results",
                   help="base folder; the outputs of each seed go to "
                        "<output-dir>/seed<N>/ (relative output paths are "
                        "joined to it); empty string to disable")
    p.add_argument("--num-seeds", type=int, default=1,
                   help="run seeds --seed, ..., --seed + num_seeds - 1 one "
                        "after the other and print a summary")
    p.add_argument("--resume-from", type=str, default=None,
                   help="continue the run saved in this checkpoint")
    p.add_argument("--compressor", choices=["zlib", "brotli"], default="zlib",
                   help="compressor of the high-order entropy (default zlib, "
                        "level 6; brotli: quality 2, as in the engine)")
    return p


def _format_path(path: Optional[str], seed: int, max_instructions: int) -> Optional[str]:
    """Fill in the {seed}/{instr}/{instr_m}/{instr_b} placeholders of an
    output path (e.g. 'seed{seed}_{instr}instr.jsonl')."""
    if path is None:
        return None
    return path.format(
        seed=seed,
        instr=max_instructions,
        instr_m=round(max_instructions / 1e6),
        instr_b=round(max_instructions / 1e9, 1),
    )


def _in_dir(path: Optional[str], dir_: Optional[str]) -> Optional[str]:
    """Join `path` to `dir_` unless `path` is absolute or either is None."""
    if path is None or not dir_:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(dir_, path)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.compressor == "brotli":
        use_brotli()

    mutation_rate = (
        args.mutation_every_n_instructions
        if args.mutation_every_n_instructions > 0
        else None
    )

    if args.checkpoint_every and not args.checkpoint_out:
        sys.exit("Error: --checkpoint-every requires --checkpoint-out.")
    if args.num_seeds > 1 and args.resume_from:
        sys.exit(
            "Error: --num-seeds > 1 and --resume-from are incompatible: "
            "a checkpoint continues one specific run."
        )

    seed_replicator = None
    if args.seed_replicator:
        if args.resume_from:
            sys.stderr.write(
                "Note: --seed-replicator is ignored with --resume-from.\n"
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

        # All the outputs of a seed go to <output-dir>/seed<N>/.
        if args.output_dir:
            seed_dir = os.path.join(args.output_dir, f"seed{seed}")
            os.makedirs(seed_dir, exist_ok=True)
        else:
            seed_dir = None

        # An empty string disables the output (None), before joining paths.
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


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soup_inspector.py

Two-phase wrapper around the cubff Python bindings that finds a
self-replicator in a large soup and then follows it in detail, while
keeping the expensive per-program self-replication check
(SimulationParams.eval_selfrep, which re-tests EVERY program in the
soup, every epoch it is on) off almost all of the time.

Why two phases
--------------
With a soup of size --num-programs (e.g. 131072), turning eval_selfrep
on for the whole run is very expensive: the engine re-runs the
self-replication test (13 noise trials per program) for ALL programs,
every single epoch, regardless of whether anything interesting is
happening. Toggling it on/off at runtime mid-simulation also turned
out to be unsafe in practice (observed corrupted scores and hangs).

So instead we split the work into two separate processes, each with
eval_selfrep fixed for its entire run (set once, never touched again):

  --phase search (cheap):
      Runs with eval_selfrep permanently OFF. Only watches the soup's
      aggregate higher-order entropy (the same criterion the paper
      itself uses for its "state transition" experiments,
      PAPER_TRANSITION_ENTROPY_THRESHOLD = 1.0 - see paper_constants.py).
      As soon as entropy crosses that threshold:
        1) it runs ONE single full-soup self-replication check (a single
           epoch with eval_selfrep momentarily set True and then
           immediately back to False - not repeated every epoch) to see
           if any program is already above --threshold;
        2) if a candidate is found, it is NOT saved immediately: for the
           next --confirm-epochs epochs we keep checking ONLY that one
           candidate program (via EvalParsedSelfrep, a cheap
           single-program call, not a full-soup pass) to make sure it is
           a real, persistent replicator and not a fluke;
        3) if the candidate falls below threshold at any point during
           confirmation, it is discarded silently (no save, no
           checkpoint) and the run goes back on its own to the cheap
           entropy-only search, ready to catch a later rise;
        4) once confirmed, the replicator's raw bytes are saved to
           --save-dir, a checkpoint of the whole soup is saved via the
           engine's own save/load mechanism (--checkpoint), and the
           confirmed program's index is saved alongside it
           (--checkpoint + ".idx"), then the process stops.

  --phase track (precise):
      A separate process. eval_selfrep is fixed OFF for the whole run
      (we don't need it: we already know which program is the
      replicator from --checkpoint + ".idx"). It reloads the checkpoint
      soup via the engine's own --load-from mechanism and, from there,
      shows every epoch how the known replicator interacts with
      whichever program it gets paired with (same before/after format
      as earlier versions of this script). The replicator's own score
      is only recomputed, via the cheap single-program EvalParsedSelfrep
      call, on epochs where something is actually being printed - never
      as a full-soup pass.

Typical usage (from the cubff/cubff folder, after building with
PYTHON=1):

    python3 -u soup_inspector.py --lang bff_noheads --num-programs 131072 \\
        --max-epochs 16000 --seed 128 --phase search --checkpoint checkpoint.dat

    python3 -u soup_inspector.py --lang bff_noheads --num-programs 131072 \\
        --max-epochs 16000 --seed 128 --phase track --checkpoint checkpoint.dat
"""

import argparse
import os
import struct
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from paper_constants import (
    PAPER_NUM_PROGRAMS,
    PAPER_DEFAULT_MUTATION_RATE,
    PAPER_SEEDED_EPOCHS,
    PAPER_TRANSITION_ENTROPY_THRESHOLD,
)

try:
    from bin import cubff
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "Could not import the compiled bin/cubff*.so module\n"
        "Make sure you:\n"
        "  1) built the project with Python bindings enabled:\n"
        "       make CUDA=0 PYTHON=1   (CPU build)\n"
        "       make PYTHON=1          (CUDA/GPU build, if you have nvcc and a GPU)\n"
        "     (by default the Makefile only builds bin/main, PYTHON=0)\n"
        "  2) have pybind11 installed: pip install pybind11\n"
        "  3) run this script from the cubff/cubff folder, or copy it\n"
        "     in there.\n"
        f"Original error: {exc}\n"
    )
    sys.exit(1)


KSINGLE_TAPE_SIZE = 64  # must match kSingleTapeSize in common.h

# Note: the engine's self-replication score (SimulationState.replication_per_prog
# / EvalParsedSelfrep) ranges 0-64, NOT 0-13. kNumIters=13 in CheckSelfRep()
# is only the number of internal noise trials used to decide whether a given
# byte position is "stable"; the final score is how many of the 2*64=128 tape
# positions turn out stable, so its true range is 0-64. kSelfrepThreshold
# (default 5, exposed below) is already expressed on that same 0-64 scale.
DEFAULT_THRESHOLD = getattr(cubff, "kSelfrepThreshold", 5)
SCORE_SCALE = KSINGLE_TAPE_SIZE  # 64: true max of the self-replication score

# Languages that use the classic "bff" operator set, with operator bytes
# equal to the ASCII codes of the characters themselves (see bff.inc.h,
# GetOpKind, when BFF_CUSTOM_OPS is not defined).
BFF_ASCII_LANGS = {"bff", "bff_noheads", "bff8", "bff8_noheads"}
BFF_OP_BYTES = {
    ord('['): '[', ord(']'): ']', ord('+'): '+', ord('-'): '-',
    ord('.'): '.', ord(','): ',', ord('<'): '<', ord('>'): '>',
    ord('{'): '{', ord('}'): '}',
}


def symbol_for_byte(b, lang):
    """Human-readable representation of a single byte, plain ASCII only."""
    if lang in BFF_ASCII_LANGS:
        if b == 0:
            return '0'
        if b in BFF_OP_BYTES:
            return BFF_OP_BYTES[b]
        return '.'  # "data" byte, not a recognized operator
    return '.'


def render_symbols(data, lang):
    return ''.join(symbol_for_byte(b, lang) for b in data)


def diff_indices(before, after):
    return [i for i, (a, b) in enumerate(zip(before, after)) if a != b]


def match_fraction(a, b):
    """Fraction of positions where the two tapes (64 bytes) match."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def describe_changes(before, after, lang, label):
    diffs = diff_indices(before, after)
    if not diffs:
        print(f"   {label}: no bytes changed")
        return
    print(f"   {label}: {len(diffs)}/{KSINGLE_TAPE_SIZE} bytes changed")
    for i in diffs:
        b_old, b_new = before[i], after[i]
        s_old = symbol_for_byte(b_old, lang)
        s_new = symbol_for_byte(b_new, lang)
        print(f"     pos {i:2d}: 0x{b_old:02x} '{s_old}'  ->  "
              f"0x{b_new:02x} '{s_new}'")


def print_tape_line(label, prog1, prog2, lang):
    s1 = render_symbols(prog1, lang)
    s2 = render_symbols(prog2, lang)
    print(f"   {label}: {s1} | {s2}")


def print_pair_interaction(epoch, pair_idx, total_pairs, p1, p2,
                            before1, before2, after1, after2, lang,
                            header_extra=""):
    print(f"\n--- epoch {epoch}, pair {pair_idx + 1}/{total_pairs}: "
          f"program #{p1}  x  program #{p2}{header_extra} ---")
    print("   (single concatenated tape: 64 bytes of prog. #%d | 64 bytes of prog. #%d)"
          % (p1, p2))
    print_tape_line("before", before1, before2, lang)
    print_tape_line("after ", after1, after2, lang)
    describe_changes(before1, after1, lang, f"program #{p1}")
    describe_changes(before2, after2, lang, f"program #{p2}")


def build_params(args):
    params = cubff.SimulationParams()
    params.num_programs = args.num_programs
    params.seed = args.seed
    params.mutation_prob = round(args.mutation_prob * (1 << 30))
    params.callback_interval = 1
    params.permute_programs = not args.no_permute
    # eval_selfrep is fixed for the WHOLE run and never toggled again once
    # RunSimulation starts (toggling it mid-run proved unsafe in practice).
    # Phase "search" needs it off almost always (only flipped on for a
    # single epoch right at the moment entropy crosses the threshold,
    # inside the callback itself - see run_search()).
    # Phase "track" never needs it at all: it already knows the
    # replicator's index from the checkpoint and checks only that one
    # program, cheaply, via EvalParsedSelfrep.
    params.eval_selfrep = False
    if args.phase == "track":
        params.load_from = args.checkpoint
    return params


def find_partner(shuffle_idx, target):
    """Find the slot index that `target` was paired with this epoch, by
    looking it up in the permutation array shuffle_idx (pairs are
    (shuffle_idx[2k], shuffle_idx[2k+1]))."""
    pos = shuffle_idx.index(target)
    if pos % 2 == 0:
        return shuffle_idx[pos + 1], pos // 2
    else:
        return shuffle_idx[pos - 1], pos // 2


def report_stats(state, prefix=""):
    print(f"{prefix}[epoch {state.epoch}] brotli_size={state.brotli_size} "
          f"brotli_bpb={state.brotli_bpb:.4f} "
          f"H0={state.h0:.4f} higher_entropy={state.higher_entropy:.6f} "
          f"total_ops={state.total_ops}")


# ---------------------------------------------------------------------------
# Phase 1: cheap entropy-only search, with a one-shot full-soup check plus
# cheap single-program confirmation once entropy crosses the threshold.
# ---------------------------------------------------------------------------

def run_search(language, params, args):
    state_holder = {
        # None -> haven't crossed the entropy threshold yet (cheap phase).
        # "checking" -> we just crossed it and asked for one full-soup pass.
        # "confirming" -> we have a candidate and are checking it cheaply,
        #                 single-program, over --confirm-epochs epochs.
        "stage": None,
        "candidate_id": None,
        "candidate_confirm_epochs": 0,
    }

    def callback(state):
        epoch = state.epoch

        if state_holder["stage"] is None:
            if state.higher_entropy >= PAPER_TRANSITION_ENTROPY_THRESHOLD:
                # Ask the engine for exactly ONE full-soup self-replication
                # pass on the NEXT epoch, then turn it back off immediately.
                params.eval_selfrep = True
                state_holder["stage"] = "checking"
                print(f"[epoch {epoch}] higher_entropy="
                      f"{state.higher_entropy:.6f} crossed "
                      f"{PAPER_TRANSITION_ENTROPY_THRESHOLD}: running a "
                      f"one-shot full-soup self-replication check...")
            elif (args.progress_every
                    and epoch % args.progress_every == 0):
                print(f"[epoch {epoch}] search phase, "
                      f"higher_entropy={state.higher_entropy:.6f} "
                      f"(target {PAPER_TRANSITION_ENTROPY_THRESHOLD})")
            return epoch >= args.max_epochs

        if state_holder["stage"] == "checking":
            # This is the epoch right after we turned eval_selfrep on, so
            # state.replication_per_prog now holds this epoch's full-soup
            # scores. Turn eval_selfrep back off immediately - we don't
            # want it computed again next epoch.
            params.eval_selfrep = False
            scores = list(state.replication_per_prog)
            candidates = [(i, s) for i, s in enumerate(scores)
                          if s >= args.threshold]
            if candidates:
                best_i, best_s = max(candidates, key=lambda t: t[1])
                state_holder["stage"] = "confirming"
                state_holder["candidate_id"] = best_i
                state_holder["candidate_confirm_epochs"] = 0
                print(f"[epoch {epoch}] candidate found: program #{best_i} "
                      f"(score {best_s}/{SCORE_SCALE}, threshold "
                      f"{args.threshold}). Confirming over the next "
                      f"{args.confirm_epochs} epochs before saving it...")
            else:
                print(f"[epoch {epoch}] false alarm: no program reached "
                      f"the threshold. Back to cheap entropy-only search.")
                state_holder["stage"] = None
            return epoch >= args.max_epochs

        # state_holder["stage"] == "confirming"
        cid = state_holder["candidate_id"]
        soup = bytes(state.soup)
        prog_bytes = soup[cid * KSINGLE_TAPE_SIZE:(cid + 1) * KSINGLE_TAPE_SIZE]
        cur_score = language.EvalParsedSelfrep(
            cubff.VectorUint8(prog_bytes), epoch, args.seed, False)
        state_holder["candidate_confirm_epochs"] += 1
        print(f"[epoch {epoch}] confirm check "
              f"{state_holder['candidate_confirm_epochs']}/"
              f"{args.confirm_epochs}: program #{cid} score now "
              f"{cur_score}/{SCORE_SCALE}")

        if cur_score < args.threshold:
            print(f"[epoch {epoch}] false alarm: program #{cid} dropped "
                  f"below threshold, discarding. Back to cheap "
                  f"entropy-only search.")
            state_holder["stage"] = None
            state_holder["candidate_id"] = None
            return epoch >= args.max_epochs

        if state_holder["candidate_confirm_epochs"] >= args.confirm_epochs:
            if args.save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = (f"replicator_{timestamp}_seed{args.seed}"
                            f"_epoch{epoch}_score{cur_score}.bin")
                save_path = os.path.join(args.save_dir, filename)
                with open(save_path, "wb") as f:
                    f.write(prog_bytes)
                print(f"   -> replicator saved to '{save_path}'")
            with open(args.checkpoint, "wb") as f:
                f.write(struct.pack("<QQQ", 1, args.num_programs, state.epoch))
                f.write(soup)
            with open(args.checkpoint + ".idx", "w") as f:
                f.write(str(cid))
            print("\n" + "=" * 70)
            print(f"*** REPLICATOR CONFIRMED: program #{cid} "
                  f"(score {cur_score}/{SCORE_SCALE}, threshold "
                  f"{args.threshold}), stable for {args.confirm_epochs} "
                  f"epochs ***")
            print(f"   checkpoint saved to '{args.checkpoint}' (index "
                  f"saved to '{args.checkpoint}.idx')")
            report_stats(state, prefix="   ")
            print("=" * 70)
            return True

        return epoch >= args.max_epochs

    language.RunSimulation(params, None, callback)


# ---------------------------------------------------------------------------
# Phase 2: precise tracking. The replicator's index is already known from
# the checkpoint - no full-soup search here at all. eval_selfrep stays off
# for the entire run; the replicator's own score is only recomputed, via a
# cheap single-program call, on epochs where we actually print something.
# ---------------------------------------------------------------------------

def run_track(language, params, args):
    with open(args.checkpoint + ".idx") as f:
        replicator_id = int(f.read().strip())
    print(f"Loaded checkpoint '{args.checkpoint}', known replicator index: "
          f"#{replicator_id}\n")

    state_holder = {
        "prev_soup": None,
        "detected_epoch": None,
        "baseline_entropy": None,
        "peak_entropy_since_detection": None,
        "extinct": False,
    }

    def callback(state):
        epoch = state.epoch
        soup = bytes(state.soup)
        prev_soup = state_holder["prev_soup"]

        if state_holder["detected_epoch"] is None:
            state_holder["detected_epoch"] = epoch
            state_holder["baseline_entropy"] = state.higher_entropy
            state_holder["peak_entropy_since_detection"] = state.higher_entropy
            state_holder["prev_soup"] = soup
            return epoch >= args.max_epochs

        detected_epoch = state_holder["detected_epoch"]
        track_limit_reached = (
            args.track_epochs is not None
            and epoch - detected_epoch > args.track_epochs
        )
        if track_limit_reached:
            return True

        baseline = state_holder["baseline_entropy"]
        peak = state_holder["peak_entropy_since_detection"]
        if state.higher_entropy > peak:
            peak = state.higher_entropy
            state_holder["peak_entropy_since_detection"] = peak

        epochs_since_detect = epoch - detected_epoch
        risen = (peak - baseline) >= args.entropy_min_rise
        fallen_back = (
            risen
            and (state.higher_entropy - baseline)
            <= args.entropy_revert_fraction * (peak - baseline)
        )
        extinct = (
            epochs_since_detect >= args.min_epochs_before_revert
            and fallen_back
        )

        if extinct:
            print("\n" + "-" * 70)
            print(f"*** Replicator #{replicator_id} appears to have gone "
                  f"extinct (epoch {epoch}): higher_entropy had risen to "
                  f"{peak:.6f} (baseline ~{baseline:.6f}) but fell back to "
                  f"{state.higher_entropy:.6f}. Stopping tracking - run "
                  f"phase 'search' again from a later point to look for a "
                  f"new replicator.")
            print("-" * 70 + "\n")
            state_holder["extinct"] = True
            return True

        show_epoch = (
            prev_soup is not None
            and epochs_since_detect % max(1, args.every) == 0
        )

        if show_epoch:
            shuffle = list(state.shuffle_idx)
            partner_id, pair_idx = find_partner(shuffle, replicator_id)
            total_pairs = len(shuffle) // 2

            r_id, p_id = replicator_id, partner_id
            b_r = prev_soup[r_id * KSINGLE_TAPE_SIZE:(r_id + 1) * KSINGLE_TAPE_SIZE]
            b_p = prev_soup[p_id * KSINGLE_TAPE_SIZE:(p_id + 1) * KSINGLE_TAPE_SIZE]
            a_r = soup[r_id * KSINGLE_TAPE_SIZE:(r_id + 1) * KSINGLE_TAPE_SIZE]
            a_p = soup[p_id * KSINGLE_TAPE_SIZE:(p_id + 1) * KSINGLE_TAPE_SIZE]

            copy_fraction = max(match_fraction(a_p, b_r),
                                 match_fraction(a_r, b_p))
            is_copy_event = copy_fraction >= args.copy_similarity

            total_changed = (len(diff_indices(b_r, a_r))
                              + len(diff_indices(b_p, a_p)))
            is_significant_mutation = total_changed >= args.min_changed_bytes
            is_notable = is_copy_event or is_significant_mutation

            cur_score = language.EvalParsedSelfrep(
                cubff.VectorUint8(a_r), epoch, args.seed, False)

            tag = ""
            if is_copy_event:
                tag = (f"  [SUCCESSFUL SELF-REPLICATION "
                       f"({copy_fraction * 100:.0f}% matching bytes)]")
            elif is_significant_mutation:
                tag = f"  [significant mutation ({total_changed} bytes)]"

            header_extra = (f"  [replicator, current score "
                             f"{cur_score}/{SCORE_SCALE}]{tag}")

            print(f"\n--- epoch {epoch}, pair {pair_idx + 1}/{total_pairs}: "
                  f"program #{r_id}  x  program #{p_id}{header_extra} ---")
            print_tape_line("before", b_r, b_p, args.lang)
            print_tape_line("after ", a_r, a_p, args.lang)

            if is_notable:
                describe_changes(b_r, a_r, args.lang, f"program #{r_id}")
                describe_changes(b_p, a_p, args.lang, f"program #{p_id}")
                if cur_score < args.threshold:
                    print(f"   (note: score dropped below threshold "
                          f"{args.threshold}; still following program "
                          f"#{replicator_id})")

        state_holder["prev_soup"] = soup
        return epoch >= args.max_epochs

    language.RunSimulation(params, None, callback)

    if state_holder["extinct"]:
        print("Simulation finished (replicator went extinct).")
    else:
        print("Simulation finished.")

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase cubff runner: 'search' cheaply watches soup "
                    "entropy and confirms a candidate replicator with "
                    "single-program checks; 'track' reloads the resulting "
                    "checkpoint (already knowing the replicator's index) "
                    "and shows its interactions epoch by epoch.")
    parser.add_argument("--phase", choices=["search", "track"], required=True,
                         help="'search': cheap entropy-only phase; confirms "
                              "a candidate and saves a checkpoint + index. "
                              "'track': loads that checkpoint and index and "
                              "follows the known replicator.")
    parser.add_argument("--checkpoint", default="checkpoint.dat",
                         help="Path to the checkpoint file written by "
                              "--phase search (soup contents) and read by "
                              "--phase track. The confirmed replicator's "
                              "index is stored alongside it as "
                              "<checkpoint>.idx (default: checkpoint.dat).")
    parser.add_argument("--confirm-epochs", type=int, default=5,
                         help="Phase 'search' only: number of epochs to "
                              "confirm a candidate replicator (cheap "
                              "single-program checks) before saving it "
                              "(default 5).")
    parser.add_argument("--lang", default="bff_noheads",
                         help="Language (default: bff_noheads). Readable "
                              "symbols work for: "
                              + ", ".join(sorted(BFF_ASCII_LANGS))
                              + "; for other languages only the hex view "
                                "is shown.")
    parser.add_argument("--num-programs", type=int, default=PAPER_NUM_PROGRAMS,
                         help=f"Number of programs in the soup (must be "
                              f"even; default {PAPER_NUM_PROGRAMS}, i.e. "
                              f"PAPER_NUM_PROGRAMS).")
    parser.add_argument("--max-epochs", type=int, default=200000,
                         help="Maximum number of epochs to run in this "
                              "phase (default 200000).")
    parser.add_argument("--track-epochs", type=int, default=None,
                         help="Phase 'track' only: how many epochs to keep "
                              "showing interactions before stopping "
                              "(default: no limit, continues until "
                              "--max-epochs).")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                         help="Minimum self-replication score (0-"
                              f"{SCORE_SCALE} scale) for a program to be "
                              f"considered a self-replicator (default: "
                              f"{DEFAULT_THRESHOLD}, same as the engine's "
                              "own cubff.kSelfrepThreshold).")
    parser.add_argument("--save", type=int, default=1, choices=[0, 1],
                     help="Phase 'search' only: if 1 (default), once a "
                          "candidate is confirmed its raw 64 bytes are "
                          "saved to a file inside --save-dir. If 0, "
                          "nothing is saved there (the checkpoint is still "
                          "written).")
    parser.add_argument("--save-dir", default="replicators",
                     help="Folder where confirmed replicators are saved "
                          "(default: 'replicators'; created if it does "
                          "not exist).")
    parser.add_argument("--every", type=int, default=1,
                         help="Phase 'track' only: show the interaction "
                              "detail only every N epochs (default 1 = "
                              "every epoch).")
    parser.add_argument("--seed", type=int, default=PAPER_SEEDED_EPOCHS)
    parser.add_argument("--mutation-prob", type=float,
                         default=PAPER_DEFAULT_MUTATION_RATE,
                         help="Background per-byte, per-epoch mutation "
                              f"probability (default {PAPER_DEFAULT_MUTATION_RATE:.6f}, "
                              "identical to the paper's default value).")
    parser.add_argument("--no-permute", action="store_true",
                         help="Disable shuffling between epochs (fixed "
                              "cyclic pairing).")
    parser.add_argument("--progress-every", type=int, default=1000,
                         help="Print a status line every N epochs while "
                              "nothing notable is happening (0 to "
                              "disable).")

    group = parser.add_argument_group(
        "Phase 'track': filter for which events are shown")
    group.add_argument("--min-changed-bytes", type=int, default=4,
                        help="Minimum total number of bytes changed "
                             "(across both programs in the pair) for an "
                             "epoch to be shown as a 'significant "
                             "mutation' (default 4).")
    group.add_argument("--copy-similarity", type=float, default=0.6,
                        help="Minimum fraction (0-1) of identical bytes "
                             "for an event to be labeled 'successful "
                             "self-replication' (default 0.6 = 60%%).")

    group2 = parser.add_argument_group(
        "Phase 'track': extinction detection")
    group2.add_argument("--entropy-min-rise", type=float, default=0.05,
                         help="How much higher_entropy must have risen "
                              "above its value at load time before we "
                              "consider a later fall a real relapse "
                              "(default 0.05).")
    group2.add_argument("--entropy-revert-fraction", type=float, default=0.3,
                         help="If, after rising, higher_entropy falls back "
                              "below baseline + this fraction of the peak "
                              "reached, the replicator is considered "
                              "extinct (default 0.3 = 30%%).")
    group2.add_argument("--min-epochs-before-revert", type=int, default=20,
                         help="Minimum tracked epochs before extinction "
                              "can be declared (default 20).")
    args = parser.parse_args()

    if args.num_programs % 2 != 0:
        args.num_programs += 1
        sys.stderr.write(
            f"num-programs must be even: rounded up to {args.num_programs}\n")

    if args.lang not in BFF_ASCII_LANGS:
        sys.stderr.write(
            f"Note: '{args.lang}' is not in the list of languages with "
            "readable symbols; only the hex view will be shown.\n")

    if args.phase == "search" and args.save:
        os.makedirs(args.save_dir, exist_ok=True)

    language = cubff.GetLanguage(args.lang)
    params = build_params(args)

    print(f"Phase: {args.phase} | language: {args.lang} | programs: "
          f"{args.num_programs} | max epochs: {args.max_epochs} | "
          f"threshold: {args.threshold}/{SCORE_SCALE} | seed: {args.seed}\n")

    cubff.ResetColors()

    if args.phase == "search":
        run_search(language, params, args)
    else:
        run_track(language, params, args)


if __name__ == "__main__":
    main()
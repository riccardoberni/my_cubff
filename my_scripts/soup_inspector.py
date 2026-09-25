#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soup_inspector.py

Wrapper around the cubff Python bindings that finds a
self-replicator in a large soup and checks that it is a real one, while
keeping the expensive per-program self-replication check
(SimulationParams.eval_selfrep, which re-tests EVERY program in the
soup, every epoch it is on) off almost all of the time.

Why the search is cheap
-----------------------
With a soup of size --num-programs (e.g. 131072), turning eval_selfrep
on for the whole run is very expensive: the engine re-runs the
self-replication test (13 noise trials per program) for ALL programs,
every single epoch, regardless of whether anything interesting is
happening. Toggling it on/off at runtime mid-simulation also turned
out to be unsafe in practice (observed corrupted scores and hangs).

So instead the search runs on its own, and following the replicator
afterwards is done by a separate process (reseed_and_track.py
--from-checkpoint):

  Search (cheap):
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
        4) once confirmed, the replicator's raw bytes and a checkpoint of
           the whole soup (via the engine's own save/load mechanism) are
           saved under --results-dir, in two subfolders ("replicators/"
           and "checkpoints/"), sharing one identifier so the two files
           belonging to the same replicator can be told apart from those
           of any other run or any other confirmed replicator. The
           confirmed program's index is saved alongside the checkpoint
           (<checkpoint>.idx). A small JSON metadata file, with the same
           identifier, records every run parameter that can affect
           whether the same --seed reproduces the same result (see
           "Where things are saved" below), then the process stops.

Where things are saved
-----------------------
By default, everything lands under --results-dir (default:
"soup_inspector_results/"):

    soup_inspector_results/
      replicators/
        replicator_<id>.bin              (the 64 raw bytes; only if --save)
        replicator_<id>.json             (run parameters; only if --save)
      checkpoints/
        checkpoint_<id>.dat              (the whole soup at save time)
        checkpoint_<id>.dat.idx          (index of the replicator in it)
        checkpoint_<id>.dat.json         (run parameters, always written,
                                           a copy of the one above)

where <id> encodes the timestamp, seed, epoch and score, so the replicator,
its checkpoint and its metadata can be paired by name even after many runs.
--checkpoint overrides the checkpoint's path (and its ".idx" sidecar) if a
fixed, non-generated location is preferred; it does not affect where the
replicator's raw bytes and metadata are saved.

<id> alone is NOT enough to tell whether two runs with the same --seed are
directly comparable: the same seed only reproduces the same soup and the
same result if every other parameter that consumes random numbers or
changes what counts as "confirmed" also matches (--num-programs,
--mutation-prob, --no-permute, --threshold, --confirm-epochs, --lang). The
JSON metadata file records exactly these, plus the number of the confirmed
program in the soup and the git commit of this script if available, so two
saved replicators can be compared properly instead of only by --seed.

Following the replicator
------------------------
Once the replicator is confirmed, the checkpoint and the index file are
used by reseed_and_track.py, which reloads that exact soup and shows every
epoch how the replicator interacts with the program it gets paired with:

    python3 -u reseed_and_track.py \\
        --from-checkpoint soup_inspector_results/checkpoints/checkpoint_<id>.dat \\
        --epochs 2000 --print-interval 1

Passing --then-track N to this script launches that command automatically
(as a separate process, for N epochs) right after the replicator has been
saved, with the correct, auto-generated checkpoint path already filled in.

Typical usage (from the cubff/cubff folder, after building with
PYTHON=1):

    python3 -u soup_inspector.py --lang bff_noheads --num-programs 131072 \\
        --max-epochs 16000 --seed 128 --checkpoint checkpoint.dat
"""

import argparse
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone

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

def build_params(args):
    params = cubff.SimulationParams()
    params.num_programs = args.num_programs
    params.seed = args.seed
    params.mutation_prob = round(args.mutation_prob * (1 << 30))
    params.callback_interval = 1
    params.permute_programs = not args.no_permute
    # eval_selfrep is off almost always (toggling it mid-run proved unsafe
    # in practice): it is only flipped on for a single epoch right at the
    # moment entropy crosses the threshold, inside the callback itself - see
    # run_search().
    params.eval_selfrep = False
    return params


def get_git_commit():
    """Short git commit hash of this script's checkout, or None if this is
    not a git checkout or git is not available. Best-effort only: never
    raises, since it is only informational metadata."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_THIS_DIR,
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def report_stats(state, prefix=""):
    print(f"{prefix}[epoch {state.epoch}] brotli_size={state.brotli_size} "
          f"brotli_bpb={state.brotli_bpb:.4f} "
          f"H0={state.h0:.4f} higher_entropy={state.higher_entropy:.6f} "
          f"total_ops={state.total_ops}")


# ---------------------------------------------------------------------------
# Search: cheap entropy-only search, with a one-shot full-soup check plus
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
        "confirmed": False,
        "checkpoint_path": None,
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
            # One identifier shared by the replicator, its checkpoint and its
            # metadata, so the files belonging to this run can be paired by
            # name (see the module docstring, "Where things are saved").
            rep_id = (f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                      f"_seed{args.seed}_epoch{epoch}_score{cur_score}")

            # <id> alone does not say whether this run is comparable to
            # another one with the same --seed: every parameter below can
            # change the outcome even at equal seed. Recorded regardless of
            # --save, since it documents the run even when only the
            # checkpoint is kept.
            metadata = {
                "id": rep_id,
                "seed": args.seed,
                "num_programs": args.num_programs,
                "mutation_prob": args.mutation_prob,
                "permute_programs": not args.no_permute,
                "lang": args.lang,
                "threshold": args.threshold,
                "confirm_epochs": args.confirm_epochs,
                "max_epochs": args.max_epochs,
                "confirmed_epoch": epoch,
                "confirmed_program_index": cid,
                "confirmed_score": cur_score,
                "score_scale": SCORE_SCALE,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": get_git_commit(),
            }

            if args.save:
                replicator_dir = os.path.join(args.results_dir, "replicators")
                os.makedirs(replicator_dir, exist_ok=True)
                replicator_path = os.path.join(
                    replicator_dir, f"replicator_{rep_id}.bin")
                with open(replicator_path, "wb") as f:
                    f.write(prog_bytes)
                with open(os.path.join(replicator_dir,
                                       f"replicator_{rep_id}.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                print(f"   -> replicator saved to '{replicator_path}'")

            if args.checkpoint is not None:
                # Explicit override: fixed path, not tied to rep_id.
                checkpoint_path = args.checkpoint
                os.makedirs(os.path.dirname(checkpoint_path) or ".",
                            exist_ok=True)
            else:
                checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_{rep_id}.dat")
            with open(checkpoint_path, "wb") as f:
                f.write(struct.pack("<QQQ", 1, args.num_programs, state.epoch))
                f.write(soup)
            with open(checkpoint_path + ".idx", "w") as f:
                f.write(str(cid))
            # Always written next to the checkpoint (which is always saved),
            # regardless of --save, so the run stays documented even when
            # the replicator's raw bytes are not kept on their own.
            with open(checkpoint_path + ".json", "w") as f:
                json.dump(metadata, f, indent=2)

            print("\n" + "=" * 70)
            print(f"*** REPLICATOR CONFIRMED: program #{cid} "
                  f"(score {cur_score}/{SCORE_SCALE}, threshold "
                  f"{args.threshold}), stable for {args.confirm_epochs} "
                  f"epochs ***")
            print(f"   checkpoint saved to '{checkpoint_path}' (index "
                  f"saved to '{checkpoint_path}.idx', run parameters saved "
                  f"to '{checkpoint_path}.json')")
            report_stats(state, prefix="   ")
            print("=" * 70)
            state_holder["confirmed"] = True
            state_holder["checkpoint_path"] = checkpoint_path
            return True

        return epoch >= args.max_epochs

    language.RunSimulation(params, None, callback)
    return state_holder["confirmed"], state_holder["checkpoint_path"]


def build_track_command(checkpoint_path, args):
    """Command that follows the saved replicator with reseed_and_track.py.
    IMPORTANT: --seed must match the seed used to generate the checkpoint,
    because the engine's per-epoch randomness is a deterministic function
    of (params.seed, epoch) - see common_language.h's `seed` lambda in
    RunSimulation. Without this, continuation diverges from the true
    trajectory starting at the very first epoch after reload - confirmed
    experimentally: with the wrong seed, the tracked replicator was
    reliably killed at the first epoch regardless of its confirmed score;
    with the correct seed (verified via bin/main --load), the same
    checkpoint showed the replicator's lineage still dominating ~96% of
    the soup (125566/131072 programs) at that same epoch.
    Also note: reseed_and_track.py runs with the engine's default mutation
    rate and shuffling; if --mutation-prob or --no-permute were changed for
    the search, the tracked continuation will not use them."""
    epochs = args.then_track if args.then_track > 0 else 2000
    return [sys.executable, "-u",
            os.path.join(_THIS_DIR, "reseed_and_track.py"),
            "--from-checkpoint", checkpoint_path,
            "--seed", str(args.seed),
            "--epochs", str(epochs),
            "--print-interval", "1",
            "--stop-on-extinction"]


def main():
    parser = argparse.ArgumentParser(
        description="cubff runner that cheaply watches soup entropy and "
                    "confirms a candidate replicator with single-program "
                    "checks, then saves it with a checkpoint of the soup. "
                    "Use reseed_and_track.py --from-checkpoint to follow it.")
    parser.add_argument("--checkpoint", default=None,
                         help="Override the checkpoint's path (advanced). "
                              "By default the checkpoint is auto-named "
                              "inside <results-dir>/checkpoints/, sharing "
                              "an identifier with the replicator saved at "
                              "the same time. If given explicitly, that "
                              "exact path is used instead (with its "
                              "'.idx' sidecar written next to it), "
                              "independently of --results-dir.")
    parser.add_argument("--confirm-epochs", type=int, default=5,
                         help="Number of epochs to "
                              "confirm a candidate replicator (cheap "
                              "single-program checks) before saving it "
                              "(default 5).")
    parser.add_argument("--lang", default="bff_noheads",
                         help="Language (default: bff_noheads).")
    parser.add_argument("--num-programs", type=int, default=PAPER_NUM_PROGRAMS,
                         help=f"Number of programs in the soup (must be "
                              f"even; default {PAPER_NUM_PROGRAMS}, i.e. "
                              f"PAPER_NUM_PROGRAMS).")
    parser.add_argument("--max-epochs", type=int, default=200000,
                         help="Maximum number of epochs to run "
                              "(default 200000).")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                         help="Minimum self-replication score (0-"
                              f"{SCORE_SCALE} scale) for a program to be "
                              f"considered a self-replicator (default: "
                              f"{DEFAULT_THRESHOLD}, same as the engine's "
                              "own cubff.kSelfrepThreshold).")
    parser.add_argument("--save", type=int, default=1, choices=[0, 1],
                     help="If 1 (default), once a "
                          "candidate is confirmed its raw 64 bytes are "
                          "saved inside --results-dir/replicators/. If 0, "
                          "nothing is saved there (the checkpoint is still "
                          "written).")
    parser.add_argument("--results-dir", default="soup_inspector_results",
                     help="Root folder for a confirmed replicator's "
                          "'replicators/' and 'checkpoints/' subfolders "
                          "(default: 'soup_inspector_results'; created if "
                          "missing). See the module docstring for the "
                          "naming scheme.")
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

    parser.add_argument("--then-track", type=int, default=0,
                         help="If > 0, right after a replicator is saved, "
                              "launch reseed_and_track.py --from-checkpoint "
                              "for this many epochs, as a separate process. "
                              "If 0 (default), only the command is printed.")
    args = parser.parse_args()

    if args.num_programs % 2 != 0:
        args.num_programs += 1
        sys.stderr.write(
            f"num-programs must be even: rounded up to {args.num_programs}\n")

    language = cubff.GetLanguage(args.lang)
    params = build_params(args)

    print(f"language: {args.lang} | programs: "
          f"{args.num_programs} | max epochs: {args.max_epochs} | "
          f"threshold: {args.threshold}/{SCORE_SCALE} | seed: {args.seed}\n")

    cubff.ResetColors()

    confirmed, checkpoint_path = run_search(language, params, args)

    if confirmed:
        cmd = build_track_command(checkpoint_path, args)
        print("\nTo follow the replicator, run:\n    " + " ".join(cmd))
        if args.then_track > 0:
            print("\nLaunching it now...\n")
            sys.stdout.flush()
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
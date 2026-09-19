import argparse
import glob
import os
import random
import struct
import sys
from bin import cubff
from run_metrics import metrics_from_state, plot_run, save_records_to_jsonl
from paper_constants import (
    PAPER_SEEDED_EPOCHS,
    PAPER_SEEDED_TRANSITION_RATE,
    PAPER_UNSEEDED_TRANSITION_RATE,
    PAPER_NUM_PROGRAMS,
    PAPER_TRANSITION_ENTROPY_THRESHOLD,
)

DEFAULT_ENTROPY_THRESHOLD = PAPER_TRANSITION_ENTROPY_THRESHOLD

LANG = "bff_noheads"

BFF_ASCII_LANGS = {"bff", "bff_noheads", "bff8", "bff8_noheads"}
BFF_OP_BYTES = {
    ord('['): '[', ord(']'): ']', ord('+'): '+', ord('-'): '-',
    ord('.'): '.', ord(','): ',', ord('<'): '<', ord('>'): '>',
}


def symbol_for_byte(b, lang):
    if lang in BFF_ASCII_LANGS:
        if b == 0:
            return '0'
        if b in BFF_OP_BYTES:
            return BFF_OP_BYTES[b]
        return '.'
    return f"{b:02x}"


def render_symbols(data, lang):
    return ''.join(symbol_for_byte(b, lang) for b in data)


def print_pair_interaction(epoch, left_id, right_id, target_id,
                            before_left, before_right, after_left, after_right, lang):
    tag_left = " [TARGET]" if left_id == target_id else ""
    tag_right = " [TARGET]" if right_id == target_id else ""
    print(f"\n--- epoch {epoch}: program #{left_id}{tag_left} x  "
          f"program #{right_id}{tag_right} ---")
    print(f"   before : {render_symbols(before_left, lang)} | {render_symbols(before_right, lang)}")
    print(f"   joined : {render_symbols(before_left + before_right, lang)}")
    print(f"   after  : {render_symbols(after_left, lang)} | {render_symbols(after_right, lang)}")


def find_latest_replicator(directory):
    """Returns the path of the most recently modified replicator*.bin file
    inside `directory`, or None if there aren't any."""
    pattern = os.path.join(directory, "replicator_*.bin")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def build_seed_soup(path, program, num_programs, target_slot, rng):
    """Writes a soup of `num_programs` programs: `program` in slot
    `target_slot`, random noise (via `rng`) everywhere else. If `program`
    is None (the "unseeded" condition), writes noise everywhere, including
    target_slot."""
    with open(path, "wb") as f:
        f.write(struct.pack("=QQQ", 0, num_programs, 0))
        for slot in range(num_programs):
            if program is not None and slot == target_slot:
                f.write(program)
            else:
                f.write(rng.randbytes(64))


def run_seeded_run(program, *, num_programs, epochs, seed, target_slot,
                    entropy_threshold, print_interval, collect_metrics,
                    soup_path):
    """Runs ONE run: places `program` in a noise soup (slot `target_slot`,
    or random if None) and runs it for `epochs` epochs. If `program` is
    None (the "unseeded" condition), the soup stays pure noise: no
    replicator is placed.

    If `print_interval` > 0, prints every so many epochs the target/partner
    pair interaction and the current entropy (useful for following one run
    "by hand"; can be turned off for statistical experiments over many
    runs). The pair detail is only printed if `program` is not None
    (otherwise there is no real target to follow).

    Returns a dict with the target_slot used, the transition epoch (or
    None), and, if `collect_metrics`, the list of per-epoch metrics for
    plot_run.
    """
    rng = random.Random(seed)
    if target_slot is None:
        target_slot = rng.randrange(num_programs)
    build_seed_soup(soup_path, program, num_programs, target_slot, rng)

    language = cubff.GetLanguage(LANG)
    params = cubff.SimulationParams()
    params.num_programs = num_programs
    params.seed = seed
    params.eval_selfrep = True
    params.load_from = soup_path
    params.callback_interval = 1

    state_holder = {"prev_soup": None, "baseline_entropy": None}
    metric_records = [] if collect_metrics else None
    found_epoch = [None]

    def callback(state):
        soup = bytes(state.soup)
        prev_soup = state_holder["prev_soup"]

        if collect_metrics:
            metric_records.append(
                metrics_from_state(state, selfrep_threshold=cubff.kSelfrepThreshold)
            )

        if state_holder["baseline_entropy"] is None:
            state_holder["baseline_entropy"] = state.higher_entropy
        baseline = state_holder["baseline_entropy"]

        if found_epoch[0] is None and state.higher_entropy > entropy_threshold:
            found_epoch[0] = state.epoch

        if print_interval and state.epoch % print_interval == 0:
            delta = state.higher_entropy - baseline
            if program is not None:
                shuffle = list(state.shuffle_idx)
                pos = shuffle.index(target_slot)
                if pos % 2 == 0:
                    left_id, right_id = target_slot, shuffle[pos + 1]
                else:
                    left_id, right_id = shuffle[pos - 1], target_slot
                partner = right_id if left_id == target_slot else left_id
                score = state.replication_per_prog[target_slot]
                print(f"epoch {state.epoch}: slot {target_slot} vs partner "
                      f"{partner}, self-rep score {score}, "
                      f"entropy={state.higher_entropy:.6f} "
                      f"(baseline={baseline:.6f}, delta={delta:+.6f})")
                if prev_soup is not None:
                    bl = prev_soup[left_id * 64:(left_id + 1) * 64]
                    br = prev_soup[right_id * 64:(right_id + 1) * 64]
                    al = soup[left_id * 64:(left_id + 1) * 64]
                    ar = soup[right_id * 64:(right_id + 1) * 64]
                    print_pair_interaction(state.epoch, left_id, right_id, target_slot,
                                            bl, br, al, ar, LANG)
            else:
                print(f"epoch {state.epoch}: (unseeded, no target) "
                      f"entropy={state.higher_entropy:.6f} "
                      f"(baseline={baseline:.6f}, delta={delta:+.6f})")

        state_holder["prev_soup"] = soup
        return state.epoch > epochs

    cubff.ResetColors()
    language.RunSimulation(params, None, callback)
    return dict(target_slot=target_slot, transition_epoch=found_epoch[0],
                metric_records=metric_records)


def main():
    parser = argparse.ArgumentParser(
        description="Takes an already-found replicator (or none, with "
                    "--unseeded), places it in a noise soup and runs it "
                    "for --runs independent runs (default 1: useful as a "
                    "quick functionality test). Always prints the status "
                    "of every run and the final transition rate, "
                    "comparable to the paper's 'seeded'/'unseeded' "
                    "experiment.")
    parser.add_argument("--unseeded", action="store_true",
                         help="If present: pure-noise soup, no replicator placed "
                              "(used to compare against the paper's 'unseeded' rate, 0.3% at 128 "
                              "epochs). --replicator/--replicator-dir are ignored. If absent "
                              "(default): the usual 'seeded' behaviour, replicator placed "
                              "in the soup.")
    parser.add_argument("--replicator", default=None,
                         help="Path to a specific replicator file "
                              "(64 raw bytes) to use. If omitted, the "
                              "most recently modified one found in "
                              "--replicator-dir is used automatically.")
    parser.add_argument("--replicator-dir", default="replicators",
                         help="Directory to search for replicators "
                              "saved by soup_inspector.py (default: "
                              "'replicators').")
    parser.add_argument("--num-programs", type=int, default=16384,
                         help="Number of programs in the soup (must be "
                              "even; default 16384; the paper uses 131072).")
    parser.add_argument("--seed", type=int, default=1,
                         help="Base seed. Run i uses seed+i: it "
                              "determines both the cubff simulation seed "
                              "and the soup noise/slot, so every run is "
                              "reproducible and independent of the "
                              "others.")
    parser.add_argument("--runs", type=int, default=1,
                         help="Number of independent runs. No behaviour "
                              "distinction based on the value: with 1 "
                              "(default) it's simply a single pass, "
                              "useful as a quick test that everything "
                              "works before launching 100.")
    parser.add_argument("--epochs", type=int, default=PAPER_SEEDED_EPOCHS,
                         help=f"Epochs per run. Default {PAPER_SEEDED_EPOCHS} "
                              "(the paper's 'seeded' protocol), independent of --runs.")
    parser.add_argument("--target-slot", type=int, default=None,
                         help="Physical slot where the replicator is placed. "
                              "Default: random on every run (as in the "
                              "paper, which places it at a random "
                              "position). Only fix it if you need to "
                              "reproduce a previous run exactly.")
    parser.add_argument("--print-interval", type=int, default=0,
                         help="Print entropy/self-rep score/pair interaction "
                              "every N epochs (0 = never, default). "
                              "Raise it (e.g. 1) to follow a run in detail.")
    parser.add_argument("--entropy-threshold", type=float,
                         default=DEFAULT_ENTROPY_THRESHOLD,
                         help="higher_entropy threshold above which a "
                              "'state transition' is considered to have "
                              "occurred. Default 1.0, as in the paper "
                              "(see Figure 6 caption: 'time to >= 1 "
                              "complexity'). Do NOT use 2.0.")
    parser.add_argument("--save-metrics", default=None,
                         help="Appends per-epoch metrics (one JSON line per epoch) of EVERY "
                              "run to this jsonl file. Independent of --runs: with --runs > 1 "
                              "the epochs of all runs end up in the same file, one after "
                              "the other.")
    parser.add_argument("--save-plot", default=None,
                         help="Saves the summary plot of the LAST run executed to this "
                              "path (with --runs > 1, it only reflects the last one).")
    parser.add_argument("--show-plot", action="store_true",
                         help="Shows the summary plot of the LAST run executed on screen "
                              "(with --runs > 1, it only reflects the last one).")
    args = parser.parse_args()

    if args.num_programs % 2 != 0:
        args.num_programs += 1
        sys.stderr.write(
            f"--num-programs must be even: rounded up to "
            f"{args.num_programs}\n")

    collect_metrics = bool(args.save_metrics or args.save_plot or args.show_plot)

    # --- BUGFIX: save_records_to_jsonl (run_metrics.py) opens its output
    # file in APPEND mode, by design, so that the per-run loop below can
    # keep adding to the same file across --runs > 1 within a single
    # invocation of this script. The downside: if this script is invoked
    # again later (e.g. after fixing a bug, or just re-running the same
    # command) with the SAME --save-metrics path, the new run silently gets
    # appended after old, stale data instead of replacing it. This produced
    # exactly that: most of the previous unseeded_128ep_run*.jsonl files
    # ended up containing several concatenated runs from different
    # invocations. Fix: truncate the file once, here, before the loop
    # starts, so every fresh invocation of reseed_and_track.py always
    # starts from a clean file, no matter how many times it has been run
    # before with this path.
    if args.save_metrics and os.path.exists(args.save_metrics):
        sys.stderr.write(
            f"Note: '{args.save_metrics}' already exists and will be "
            f"truncated before this run (use a new --save-metrics path per "
            f"experiment if you want to keep old data around).\n")
        open(args.save_metrics, "w").close()

    # --- pick which replicator to use (only if not --unseeded) ------------
    if args.unseeded:
        program = None
        replicator_path = None
    else:
        if args.replicator:
            replicator_path = args.replicator
            if not os.path.isfile(replicator_path):
                sys.exit(f"Error: replicator file not found: "
                          f"'{replicator_path}'")
        else:
            replicator_path = find_latest_replicator(args.replicator_dir)
            if replicator_path is None:
                sys.exit(f"Error: no 'replicator_*.bin' file found in "
                          f"'{args.replicator_dir}/'. Generate a replicator with "
                          f"soup_inspector.py, or specify an explicit "
                          f"file with --replicator.")
            print(f"No --replicator specified: using the most recent "
                  f"one found in '{args.replicator_dir}/': '{replicator_path}'")

        program = open(replicator_path, "rb").read()
        if len(program) != 64:
            sys.exit(f"Error: '{replicator_path}' contains {len(program)} "
                      f"bytes instead of 64.")

    # --- single loop: identical for --runs 1 or --runs > 1 -----------------
    all_results = []
    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        soup_path = f"_run_{seed}.dat"
        result = run_seeded_run(
            program,
            num_programs=args.num_programs,
            epochs=args.epochs,
            seed=seed,
            target_slot=args.target_slot,
            entropy_threshold=args.entropy_threshold,
            print_interval=args.print_interval,
            collect_metrics=collect_metrics,
            soup_path=soup_path,
        )
        all_results.append(result)

        status = (f"transition at epoch {result['transition_epoch']}"
                   if result['transition_epoch'] is not None
                   else "no transition")
        print(f"[{run_idx + 1}/{args.runs}] slot {result['target_slot']}: {status}")
        if os.path.exists(soup_path):
            os.remove(soup_path)

        if args.save_metrics and result["metric_records"]:
            save_records_to_jsonl(result["metric_records"], args.save_metrics)

    n_transitions = sum(1 for r in all_results if r["transition_epoch"] is not None)
    rate = n_transitions / args.runs
    print(f"\nTransition rate: {rate:.1%} ({n_transitions}/{args.runs}) "
          f"over {args.epochs} epochs, num_programs={args.num_programs}.")
    ref_rate = PAPER_UNSEEDED_TRANSITION_RATE if args.unseeded else PAPER_SEEDED_TRANSITION_RATE
    ref_label = "unseeded" if args.unseeded else "seeded"
    print(f"Paper reference ({ref_label}, {PAPER_SEEDED_EPOCHS} epochs, "
          f"num_programs={PAPER_NUM_PROGRAMS}): ~{ref_rate:.1%}.")
    if args.num_programs != PAPER_NUM_PROGRAMS:
        print(f"Note: --num-programs differs from the paper's ({PAPER_NUM_PROGRAMS}): "
              "the direct rate comparison is less meaningful.")
    if args.epochs != PAPER_SEEDED_EPOCHS:
        print(f"Note: --epochs differs from the paper's ({PAPER_SEEDED_EPOCHS}): "
              "the direct rate comparison is less meaningful.")

    if args.save_metrics:
        print(f"Metrics for all runs saved to '{args.save_metrics}'.")

    if (args.save_plot or args.show_plot) and all_results[-1]["metric_records"]:
        last = all_results[-1]
        title_src = os.path.basename(replicator_path) if replicator_path else "unseeded (pure noise)"
        plot_run(
            last["metric_records"],
            title=f"reseed_and_track: {title_src} "
                  f"(slot {last['target_slot']}, last of {args.runs} runs)",
            entropy_threshold=args.entropy_threshold,
            save_path=args.save_plot,
            show=args.show_plot,
        )


if __name__ == "__main__":
    main()
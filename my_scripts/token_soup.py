#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
token_soup.py

Tracer-token analysis of one soup run (paper Section 2.1, "How
self-replicators emerge: a case study", Fig. 1).

The cubff engine does not track where each byte of the soup comes from, so
this script re-runs the same simulation on the CPU with an extra array of
"origin" ids, one per byte (bff_tokens.c, loaded through ctypes):

  * a byte created at initialization or by a mutation gets a new origin id;
  * the copy instructions '.' and ',' copy the origin id with the byte;
  * '+' and '-' change the byte but keep its origin id.

As in the paper, a token is the pair (origin, current character).

The engine is run at the same time, with the same seed, and is the
reference: every --sample-interval epochs the soup computed here is compared
byte by byte with the engine's soup, and the run stops at the first
difference. The high-order entropy written to the output is the engine's.

For every sampled epoch the script writes
  <out-dir>/seed<N>_stats.csv   epoch, unique tokens, number of soup bytes
                                 holding one of the 32 most frequent tokens,
                                 number of zero bytes, entropies;
  <out-dir>/seed<N>_top32.csv   the 32 most frequent tokens and their counts.

Usage (from the cubff folder, after building with PYTHON=1):
    python3 -u my_scripts/token_soup.py --seed 171 --max-epochs 5000
"""

import argparse
import csv
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

TAPE_SIZE = 64            # bytes per program (kSingleTapeSize in cubff)
MAX_STEPS = 8 * 1024      # step limit per interaction, as in the engine
POSITION_BITS = 32        # must match bff_tokens.c
MASK64 = (1 << 64) - 1

STATS_FIELDS = ["epoch", "unique_tokens", "top32_tokens", "zeros",
                "higher_entropy", "h0", "brotli_bpb", "soup_matches_engine"]
TOP_FIELDS = ["epoch", "rank", "count", "origin_epoch", "origin_program",
              "origin_offset", "char"]


def import_cubff():
    try:
        from bin import cubff
        return cubff
    except ImportError as exc:
        sys.stderr.write(
            "Could not import bin/cubff*.so. Build with PYTHON=1 and run "
            f"from the cubff folder.\nOriginal error: {exc}\n")
        sys.exit(1)


def load_token_library():
    """Compile bff_tokens.c if needed and load it."""
    src = os.path.join(_THIS_DIR, "bff_tokens.c")
    lib_path = os.path.join(_THIS_DIR, "libbff_tokens.so")
    if (not os.path.exists(lib_path)
            or os.path.getmtime(lib_path) < os.path.getmtime(src)):
        cmd = ["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", "-o", lib_path,
               src]
        print("compiling:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(lib_path)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.ts_shuffle.argtypes = [u32p, ctypes.c_uint64, ctypes.c_uint64,
                               ctypes.c_uint64]
    lib.ts_shuffle.restype = None
    lib.ts_run_epoch.argtypes = [u8p, u64p, u32p, ctypes.c_uint64,
                                 ctypes.c_uint64, ctypes.c_uint64,
                                 ctypes.c_uint32, ctypes.c_uint64]
    lib.ts_run_epoch.restype = None
    return lib


def splitmix64(x):
    """Same generator as SplitMix64() in cubff's common_language.h."""
    z = (x + 0x9e3779b97f4a7c15) & MASK64
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK64
    return z ^ (z >> 31)


def engine_initial_soup(seed, num_programs):
    """The random soup the engine creates for this seed, byte for byte
    (InitPrograms in common_language.h)."""
    init_seed = splitmix64(splitmix64(seed) ^ splitmix64(0))
    base = (TAPE_SIZE * num_programs * init_seed) & MASK64
    with np.errstate(over="ignore"):  # 64-bit wrap-around is intended
        z = np.arange(TAPE_SIZE * num_programs, dtype=np.uint64)
        z = z + np.uint64(base)
        z = z + np.uint64(0x9e3779b97f4a7c15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(0xFF)).astype(np.uint8)


class TokenSoup:
    """The soup and its origin ids, advanced one epoch at a time."""

    def __init__(self, lib, seed, num_programs, mutation_raw):
        self.lib = lib
        self.seed = seed
        self.n = num_programs
        self.mutation_raw = mutation_raw
        self.soup = np.ascontiguousarray(
            engine_initial_soup(seed, num_programs))
        # Initial tokens: origin epoch 0, origin position = soup position.
        self.origin = np.arange(TAPE_SIZE * num_programs, dtype=np.uint64)
        self.pairing = np.zeros(num_programs, dtype=np.uint32)
        self.epochs_done = 0

    def _ptr(self, array, ctype):
        return array.ctypes.data_as(ctypes.POINTER(ctype))

    def step(self):
        """Run one epoch (pairing, mutation, execution)."""
        epoch = self.epochs_done
        self.lib.ts_shuffle(self._ptr(self.pairing, ctypes.c_uint32), self.n,
                            self.seed, epoch)
        self.lib.ts_run_epoch(self._ptr(self.soup, ctypes.c_uint8),
                              self._ptr(self.origin, ctypes.c_uint64),
                              self._ptr(self.pairing, ctypes.c_uint32),
                              self.n, self.seed, epoch, self.mutation_raw,
                              MAX_STEPS)
        self.epochs_done += 1

    def token_counts(self):
        """Distinct tokens (origin, current character) and their counts."""
        keys = (self.origin << np.uint64(8)) | self.soup.astype(np.uint64)
        return np.unique(keys, return_counts=True)


def describe_token(key):
    """Token key -> (origin epoch, origin program, origin offset, char)."""
    key = int(key)
    char = key & 0xFF
    origin = key >> 8
    position = origin & ((1 << POSITION_BITS) - 1)
    return (origin >> POSITION_BITS, position // TAPE_SIZE,
            position % TAPE_SIZE, char)


def main():
    parser = argparse.ArgumentParser(
        description="Tracer-token statistics of one soup run (paper Fig. 1).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--sample-interval", type=int, default=16,
                        help="Epochs between two samples (and two checks "
                             "against the engine).")
    parser.add_argument("--num-programs", type=int, default=131072)
    parser.add_argument("--mutation-prob", type=float, default=2 ** -12,
                        help="Per-byte mutation probability (default 2^-12, "
                             "the paper's 0.024%%).")
    parser.add_argument("--lang", default="bff_noheads")
    parser.add_argument("--out-dir", default="soup_inspector_results/tokens")
    args = parser.parse_args()

    if args.lang != "bff_noheads":
        sys.exit("bff_tokens.c implements the bff_noheads language only")
    if args.num_programs % 2:
        sys.exit("--num-programs must be even")
    if TAPE_SIZE * args.num_programs >= 1 << POSITION_BITS:
        sys.exit("--num-programs too large for the origin id layout")

    cubff = import_cubff()
    lib = load_token_library()
    mutation_raw = round(args.mutation_prob * (1 << 30))
    language = cubff.GetLanguage(args.lang)
    if hasattr(cubff, "ResetColors"):
        cubff.ResetColors()

    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, f"seed{args.seed}_stats.csv")
    top_path = os.path.join(args.out_dir, f"seed{args.seed}_top32.csv")

    sim = TokenSoup(lib, args.seed, args.num_programs, mutation_raw)
    total_bytes = TAPE_SIZE * args.num_programs
    print(f"seed={args.seed} programs={args.num_programs} "
          f"mutation={mutation_raw}/2^30 max_epochs={args.max_epochs} "
          f"sample_interval={args.sample_interval} | "
          f"{total_bytes} tokens at initialization")

    params = cubff.SimulationParams()
    params.num_programs = args.num_programs
    params.seed = args.seed
    params.mutation_prob = mutation_raw
    params.callback_interval = args.sample_interval
    params.permute_programs = True
    params.eval_selfrep = False

    t0 = time.time()
    with open(stats_path, "w", newline="") as fs, \
            open(top_path, "w", newline="") as ft:
        stats = csv.DictWriter(fs, fieldnames=STATS_FIELDS)
        top = csv.DictWriter(ft, fieldnames=TOP_FIELDS)
        stats.writeheader()
        top.writeheader()

        def callback(state):
            # state.epoch = number of epochs the engine has executed so far.
            while sim.epochs_done < state.epoch:
                sim.step()
            same = sim.soup.tobytes() == bytes(state.soup)

            keys, counts = sim.token_counts()
            order = np.argsort(counts)[::-1][:32]
            stats.writerow({
                "epoch": state.epoch,
                "unique_tokens": len(keys),
                "top32_tokens": int(counts[order].sum()),
                "zeros": int(np.count_nonzero(sim.soup == 0)),
                "higher_entropy": f"{state.higher_entropy:.6f}",
                "h0": f"{state.h0:.6f}",
                "brotli_bpb": f"{state.brotli_bpb:.6f}",
                "soup_matches_engine": int(same),
            })
            for rank, k in enumerate(order, 1):
                o_epoch, o_prog, o_off, char = describe_token(keys[k])
                top.writerow({"epoch": state.epoch, "rank": rank,
                              "count": int(counts[k]),
                              "origin_epoch": o_epoch,
                              "origin_program": o_prog,
                              "origin_offset": o_off, "char": char})
            fs.flush()
            ft.flush()
            print(f"epoch {state.epoch:6d}: unique tokens {len(keys):8d}, "
                  f"top32 {int(counts[order].sum()):8d}, entropy "
                  f"{state.higher_entropy:.3f}, soup "
                  f"{'identical to' if same else 'DIFFERENT from'} the "
                  f"engine's ({time.time() - t0:.0f} s)")
            if not same:
                print("The token simulation diverged from the engine: "
                      "stopping.")
                return True
            return state.epoch >= args.max_epochs

        language.RunSimulation(params, None, callback)

    print(f"written {stats_path} and {top_path}")


if __name__ == "__main__":
    main()
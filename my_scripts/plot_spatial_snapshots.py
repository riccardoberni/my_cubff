#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spatial_snapshots.py

Images of the spatial soup from the snapshots written by spatial_survey.py
(snapshots/seed<N>_epoch<E>.npz), drawn as in cubff's --draw_to_2d and in
Fig. 8 of the paper: every program is an 8 x 8 square of pixels, one pixel
per byte (row-major), and the colour of a pixel depends on the byte.

Two palettes are available:
  engine  the colours of the current cubff (InitByteColors in bff.inc.h);
          with this palette the full image is identical to the frame that
          bin/main --draw_to_2d writes for the same soup (--check-ppm).
  paper   the colours of the paper's Fig. 8, read from the pixels of the
          figure: brackets green, + - . , magenta, < > { } blue, the zero
          byte orange, every other byte grey (same greys as the engine).
In both palettes the first row and column of every square are darkened by
32, as in the engine, to make the borders between programs visible.

Modes:
  image   one PNG of the whole grid or of a crop (--crop).
  panels  one figure with several epochs of the same seed side by side.
  fig8    the layout of the paper's Fig. 8: a crop of the grid on the left
          and, on the right, the bytes of a column of programs of the crop
          printed as text, joined to their squares by lines.
  auto    for one seed: panels of the grid at the snapshots closest to 5%,
          25%, 50% and 75% of self-replicators (top row: bytes; bottom row:
          programs marked as self-replicators), and the Fig. 8 layout at the
          snapshot closest to 50%, with a 30 x 40 crop crossed by the front
          of the wave chosen automatically.

Typical usage:
    python3 my_scripts/plot_spatial_snapshots.py image \
        spatial_results/grid_240x135/snapshots/seed1_epoch8193.npz \
        --out images/grid_seed1_epoch8193.png
    python3 my_scripts/plot_spatial_snapshots.py panels \
        --snapshot-dir spatial_results/grid_240x135/snapshots --seed 1 \
        --epochs 1025,2049,3073,4097 --out images/fig8_panels.pdf
    python3 my_scripts/plot_spatial_snapshots.py fig8 \
        spatial_results/grid_240x135/snapshots/seed1_epoch8193.npz \
        --crop 100 40 30 40 --text-column 20 --text-rows 10 30 \
        --out images/fig8_ours.pdf
"""

# Epochs are labelled as in spatial_survey.py and in its traces: the number of
# epochs executed when the snapshot was taken (the engine's state.epoch).

import argparse
import os

import numpy as np

TAPE = 64
SIDE = 8  # a program is drawn as an 8 x 8 square

COMMANDS = "[]+-.,<>{}"


def palette(name):
    """256 x 3 uint8 array: the colour of every byte value."""
    colors = np.zeros((256, 3), dtype=np.uint8)
    for b in range(256):
        v = 192 + b // 4
        colors[b] = (v, v, v)
    if name == "engine":
        groups = {"[]": (0, 192, 0), "+-.,": (200, 0, 200),
                  "<>{}": (200, 128, 220)}
        zero = (255, 0, 0)
    elif name == "paper":
        groups = {"[]": (0, 192, 0), "+-.,": (200, 0, 200),
                  "<>{}": (0, 128, 220)}
        zero = (255, 64, 0)
    else:
        raise ValueError(name)
    for chars, rgb in groups.items():
        for c in chars:
            colors[ord(c)] = rgb
    colors[0] = zero
    return colors


def render(soup, colors):
    """soup: (rows, cols, 64) uint8 -> RGB image (rows*8, cols*8, 3)."""
    rows, cols, _ = soup.shape
    img = colors[soup].reshape(rows, cols, SIDE, SIDE, 3)
    img = img.astype(np.int16)
    img[:, :, 0, :, :] -= 32  # first row of every square
    img[:, :, 1:, 0, :] -= 32  # first column (corner darkened once)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img.transpose(0, 2, 1, 3, 4).reshape(rows * SIDE, cols * SIDE, 3)


def load(path):
    data = np.load(path)
    return data["soup"], data["selfrep"], int(data["epoch"])


def crop(soup, spec):
    if spec is None:
        return soup
    col, row, ncols, nrows = spec
    return soup[row:row + nrows, col:col + ncols]


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(maxsplit=4)
    assert parts[0] == b"P6" and parts[3] == b"255"
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(parts[4][:w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def byte_char(b):
    """Text representation of a byte, as in the paper's Fig. 8: commands as
    themselves, the zero byte as '0', other printable Latin-1 characters as
    themselves and the rest as a space."""
    if chr(b) in COMMANDS:
        return chr(b)
    if b == 0:
        return "0"
    if 0x21 <= b <= 0x7e or 0xa1 <= b <= 0xff:
        return chr(b)
    return " "


def cmd_image(args):
    from PIL import Image
    soup, _, epoch = load(args.snapshot)
    img = render(crop(soup, args.crop), palette(args.palette))
    if args.check_ppm:
        ref = read_ppm(args.check_ppm)
        full = render(soup, palette("engine"))
        same = ref.shape == full.shape and np.array_equal(ref, full)
        print(f"engine palette vs {args.check_ppm}: "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
    if args.scale > 1:
        img = img.repeat(args.scale, axis=0).repeat(args.scale, axis=1)
    Image.fromarray(img).save(args.out)
    print(f"epoch {epoch}: wrote {args.out}")


def cmd_panels(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [int(e) for e in args.epochs.split(",")]
    colors = palette(args.palette)
    fig, axes = plt.subplots(1, len(epochs),
                             figsize=(args.panel_width * len(epochs),
                                      args.panel_width * 0.62 + 0.4))
    axes = np.atleast_1d(axes)
    for ax, epoch in zip(axes, epochs):
        path = os.path.join(args.snapshot_dir,
                            f"seed{args.seed}_epoch{epoch}.npz")
        soup, _, _ = load(path)
        ax.imshow(render(crop(soup, args.crop), colors),
                  interpolation="nearest")
        ax.set_title(f"epoch {epoch}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


def draw_fig8(soup, colors, crop_spec, text_col, first, count, out,
              width=12.0, font_size=4.5, dpi=300):
    """Layout of the paper's Fig. 8: a crop of the grid on the left and the
    bytes of `count` programs of column `text_col` of the crop, starting at
    row `first`, printed as text on the right and joined to their squares."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle

    part = crop(soup, crop_spec)
    fig = plt.figure(figsize=(width, width * 0.55))
    ax_img = fig.add_axes([0.0, 0.0, 0.42, 1.0])
    ax_txt = fig.add_axes([0.47, 0.0, 0.53, 1.0])
    ax_img.imshow(render(part, colors), interpolation="nearest")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.add_patch(Rectangle((text_col * SIDE - 0.5, first * SIDE - 0.5),
                               SIDE, count * SIDE, fill=False, lw=0.6,
                               edgecolor="black"))
    ax_txt.set_xlim(0, TAPE)
    ax_txt.set_ylim(count, 0)
    ax_txt.axis("off")
    hex_colors = ["#%02x%02x%02x" % tuple(c) for c in colors]
    for k in range(count):
        tape = part[first + k, text_col]
        for j, b in enumerate(tape):
            ch = byte_char(int(b))
            if ch == " ":
                continue
            grey = colors[b][0] == colors[b][1] == colors[b][2]
            ax_txt.text(j + 0.5, k + 0.5, ch, family="monospace",
                        fontsize=font_size, ha="center", va="center",
                        color="black" if grey else hex_colors[b])
        line = ConnectionPatch(
            xyA=((text_col + 1) * SIDE - 0.5, (first + k + 0.5) * SIDE),
            coordsA=ax_img.transData, xyB=(0, k + 0.5),
            coordsB=ax_txt.transData, lw=0.3, color="black")
        fig.add_artist(line)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def cmd_fig8(args):
    soup, _, epoch = load(args.snapshot)
    first, count = args.text_rows
    draw_fig8(soup, palette(args.palette), args.crop, args.text_column,
              first, count, args.out, args.width, args.font_size, args.dpi)
    print(f"epoch {epoch}: wrote {args.out}")


def snapshot_series(snapshot_dir, seed):
    """(epoch, path, fraction of self-replicators) for every snapshot of a
    seed, sorted by epoch."""
    import glob
    import re
    series = []
    for path in glob.glob(os.path.join(snapshot_dir, f"seed{seed}_epoch*.npz")):
        epoch = int(re.search(r"_epoch(\d+)\.npz$", path).group(1))
        selfrep = np.load(path)["selfrep"]
        series.append((epoch, path, float((selfrep >= 5).mean())))
    return sorted(series)


def best_window(selfrep, ncols, nrows, target=0.5):
    """Top-left corner (col, row) of the ncols x nrows window of the grid
    whose fraction of self-replicators is closest to `target`, i.e. a window
    crossed by the front of the wave."""
    rows, cols = selfrep.shape
    mask = (selfrep >= 5).astype(np.int64)
    integral = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    integral[1:, 1:] = mask.cumsum(0).cumsum(1)
    best, best_score = (0, 0), None
    for r in range(rows - nrows + 1):
        for c in range(cols - ncols + 1):
            total = (integral[r + nrows, c + ncols] - integral[r, c + ncols]
                     - integral[r + nrows, c] + integral[r, c])
            score = abs(total / (ncols * nrows) - target)
            if best_score is None or score < best_score:
                best, best_score = (c, r), score
    return best


def cmd_auto(args):
    """Panels over time and Fig. 8 layout for one seed, choosing the epochs
    and the crop from the self-replicator maps of the snapshots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = snapshot_series(args.snapshot_dir, args.seed)
    if not series:
        raise SystemExit(f"no snapshots of seed {args.seed} in "
                         f"{args.snapshot_dir}")
    for epoch, _, fraction in series:
        print(f"snapshot epoch {epoch}: self-replicators {100 * fraction:.1f}%")
    colors = palette(args.palette)

    # Panels: the snapshots closest to the target fractions.
    chosen = []
    for target in (0.05, 0.25, 0.5, 0.75):
        item = min(series, key=lambda s: abs(s[2] - target))
        if item not in chosen:
            chosen.append(item)
    chosen.sort()
    # Two rows: the bytes of the grid, as in the paper's Fig. 8, and the map
    # of the programs that the engine's check marks as self-replicators.
    fig, axes = plt.subplots(2, len(chosen),
                             figsize=(3.2 * len(chosen), 2 * 3.2 * 0.62 + 0.5))
    axes = np.array(axes).reshape(2, len(chosen))
    selfrep_colors = np.array([[233, 233, 233], [42, 120, 214]], dtype=np.uint8)
    for k, (epoch, path, fraction) in enumerate(chosen):
        soup, selfrep, _ = load(path)
        axes[0, k].imshow(render(soup, colors), interpolation="nearest")
        axes[0, k].set_title(f"epoch {epoch} ({100 * fraction:.0f}%)",
                             fontsize=9)
        axes[1, k].imshow(selfrep_colors[(selfrep >= 5).astype(int)],
                          interpolation="nearest")
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("bytes", fontsize=9)
    axes[1, 0].set_ylabel("self-replicators", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_panels, dpi=args.dpi)
    plt.close(fig)
    print("panels: epochs " + ", ".join(str(e) for e, _, _ in chosen)
          + f" -> {args.out_panels}")

    # Fig. 8 layout: the snapshot closest to 50% of self-replicators, a
    # 30 x 40 crop crossed by the front, and the column of the crop closest
    # to half self-replicators printed as text.
    epoch, path, fraction = min(series, key=lambda s: abs(s[2] - 0.5))
    soup, selfrep, _ = load(path)
    ncols, nrows = min(30, selfrep.shape[1]), min(40, selfrep.shape[0])
    col, row = best_window(selfrep, ncols, nrows)
    window = selfrep[row:row + nrows, col:col + ncols] >= 5
    text_col = int(np.argmin(np.abs(window.mean(axis=0) - 0.5)))
    count = min(30, nrows)
    first = (nrows - count) // 2
    draw_fig8(soup, colors, (col, row, ncols, nrows), text_col, first, count,
              args.out_fig8, dpi=args.dpi)
    print(f"fig8: epoch {epoch} ({100 * fraction:.1f}% self-replicators), "
          f"crop columns {col}-{col + ncols - 1}, rows {row}-{row + nrows - 1}, "
          f"text: column {col + text_col}, rows {row + first}-"
          f"{row + first + count - 1} -> {args.out_fig8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--palette", choices=["paper", "engine"],
                       default="paper")
        p.add_argument("--crop", type=int, nargs=4, default=None,
                       metavar=("COL", "ROW", "NCOLS", "NROWS"),
                       help="Part of the grid to draw, in programs.")
        p.add_argument("--out", required=True)
        p.add_argument("--dpi", type=int, default=300)

    p = sub.add_parser("image")
    p.add_argument("snapshot")
    common(p)
    p.add_argument("--scale", type=int, default=1,
                   help="Enlarge every pixel to scale x scale pixels.")
    p.add_argument("--check-ppm", default=None,
                   help="Frame written by bin/main --draw_to_2d for the same "
                        "soup: check that the engine palette gives the same "
                        "image.")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("panels")
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", required=True,
                   help="Comma-separated snapshot epochs (as in the file "
                        "names).")
    p.add_argument("--panel-width", type=float, default=3.2)
    common(p)
    p.set_defaults(func=cmd_panels)

    p = sub.add_parser("fig8")
    p.add_argument("snapshot")
    common(p)
    p.add_argument("--text-column", type=int, required=True,
                   help="Column of the crop whose programs are printed.")
    p.add_argument("--text-rows", type=int, nargs=2, required=True,
                   metavar=("FIRST", "COUNT"),
                   help="Rows of the crop whose programs are printed.")
    p.add_argument("--width", type=float, default=12.0)
    p.add_argument("--font-size", type=float, default=4.5)
    p.set_defaults(func=cmd_fig8)

    p = sub.add_parser("auto", help="Panels over time and Fig. 8 layout for "
                                     "one seed, chosen automatically.")
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--palette", choices=["paper", "engine"], default="paper")
    p.add_argument("--out-panels", required=True)
    p.add_argument("--out-fig8", required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(func=cmd_auto)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
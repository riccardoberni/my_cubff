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
        ax.set_title(f"epoch {epoch - 1}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


def cmd_fig8(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle

    soup, _, epoch = load(args.snapshot)
    colors = palette(args.palette)
    part = crop(soup, args.crop)
    rows, cols, _ = part.shape
    text_col = args.text_column
    first, count = args.text_rows

    fig = plt.figure(figsize=(args.width, args.width * 0.55))
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
            ax_txt.text(j + 0.5, k + 0.5, ch, family="monospace",
                        fontsize=args.font_size, ha="center", va="center",
                        color="black" if colors[b][0] == colors[b][1] == colors[b][2]
                        else hex_colors[b])
        line = ConnectionPatch(
            xyA=((text_col + 1) * SIDE - 0.5, (first + k + 0.5) * SIDE),
            coordsA=ax_img.transData, xyB=(0, k + 0.5),
            coordsB=ax_txt.transData, lw=0.3, color="black")
        fig.add_artist(line)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"epoch {epoch}: wrote {args.out}")


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
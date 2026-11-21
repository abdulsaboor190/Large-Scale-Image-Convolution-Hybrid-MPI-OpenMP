#!/usr/bin/env python3
"""
Generate large synthetic benchmark images (8-bit grayscale PGM, P5).

The image is written in row chunks, so memory stays small even at 20000x20000
(400 MB on disk). Content is deterministic for a given --seed and has
gradients, rings, a checkerboard and noise, so every filter has something
visible to act on.

Examples
  # one big image for the main benchmark
  python3 scripts/gen_dataset.py --size 20000 --out data/large/big_20000.pgm

  # non-square
  python3 scripts/gen_dataset.py --width 30000 --height 12000 --out data/large/wide.pgm

  # batch of 2000 images of 2048x2048 (batch-mode experiment)
  python3 scripts/gen_dataset.py --count 2000 --size 2048 --out data/large/batch/
"""
import argparse
import os
import numpy as np

from pgmio import pgm_header

CHUNK = 256


def make_rows(y0, y1, w, h, seed):
    rng = np.random.default_rng([seed, y0])
    y = (np.arange(y0, y1, dtype=np.float32) / h)[:, None]
    x = (np.arange(w, dtype=np.float32) / w)[None, :]
    base = 0.5 + 0.25 * np.sin(2 * np.pi * (3 * x + 2 * y))
    r = np.hypot(x - 0.5, (y - 0.5) * (h / w))
    rings = (np.sin(2 * np.pi * 40 * r) > 0) * 0.25
    cell = max(w, h) // 40 or 1
    ys = np.arange(y0, y1)[:, None] // cell
    xs = np.arange(w)[None, :] // cell
    checker = ((xs + ys) % 2 == 0) * 0.12
    noise = rng.normal(0, 0.02, size=(y1 - y0, w)).astype(np.float32)
    img = np.clip(base + rings + checker + noise, 0, 1)
    return (img * 255 + 0.5).astype(np.uint8)


def write_image(path, w, h, seed):
    with open(path, "wb") as f:
        f.write(pgm_header(w, h))
        for y0 in range(0, h, CHUNK):
            f.write(make_rows(y0, min(y0 + CHUNK, h), w, h, seed).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, help="square image side in pixels")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--count", type=int, default=1, help="number of images (batch mode if > 1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="file (single image) or directory (batch)")
    a = ap.parse_args()

    w = a.width or a.size
    h = a.height or a.size
    if not w or not h:
        ap.error("give --size, or both --width and --height")

    if a.count > 1:
        os.makedirs(a.out, exist_ok=True)
        for i in range(a.count):
            write_image(os.path.join(a.out, f"img_{i:05d}.pgm"), w, h, a.seed + i)
            if (i + 1) % 100 == 0 or i + 1 == a.count:
                print(f"  {i + 1}/{a.count}", flush=True)
        print(f"wrote {a.count} images of {w}x{h} to {a.out}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        write_image(a.out, w, h, a.seed)
        print(f"wrote {a.out}: {w}x{h}, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

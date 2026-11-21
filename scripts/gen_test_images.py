#!/usr/bin/env python3
"""
Create the small correctness-test set in data/test/:
  inputs        data/test/<name>.pgm
  expected out  data/test/ref/<name>__<filter>_<k>x<k>.pgm   (from refconv.py)

Images are chosen to catch specific bugs:
  impulse_33   single bright pixel: output shows the kernel itself
  flat_64      constant image: blur/sharpen must leave it unchanged, Sobel must give 0
  step_64      vertical step edge: Sobel must respond only at the edge
  shapes_256   general content
  odd_101x77   dimensions that do not divide evenly by 2/4/8: tests MPI row-partition remainders
  tiny_5x5     smaller than the 7x7 kernel: tests border handling
"""
import os
import numpy as np

from pgmio import write_pgm
import refconv

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "test")
FILTERS = ["sobel", "gaussian", "sharpen"]
SIZES = [3, 5, 7]


def images():
    imgs = {}
    a = np.zeros((33, 33), np.uint8)
    a[16, 16] = 255
    imgs["impulse_33"] = a
    imgs["flat_64"] = np.full((64, 64), 128, np.uint8)
    s = np.zeros((64, 64), np.uint8)
    s[:, 32:] = 200
    imgs["step_64"] = s

    yy, xx = np.mgrid[0:256, 0:256]
    sh = (xx * 255 // 255).astype(np.float64) * 0.3
    sh[(xx - 100) ** 2 + (yy - 120) ** 2 < 50 ** 2] = 220
    sh[40:90, 150:230] = 90
    sh += np.random.default_rng(0).normal(0, 4, sh.shape)
    imgs["shapes_256"] = np.clip(sh, 0, 255).astype(np.uint8)

    rng = np.random.default_rng(1)
    imgs["odd_101x77"] = rng.integers(0, 256, (77, 101), dtype=np.uint8)
    imgs["tiny_5x5"] = rng.integers(0, 256, (5, 5), dtype=np.uint8)
    return imgs


def main():
    os.makedirs(os.path.join(OUT, "ref"), exist_ok=True)
    n = 0
    for name, img in images().items():
        write_pgm(os.path.join(OUT, f"{name}.pgm"), img)
        for f in FILTERS:
            for k in SIZES:
                write_pgm(os.path.join(OUT, "ref", f"{name}__{f}_{k}x{k}.pgm"), refconv.apply(img, f, k))
                n += 1
    print(f"wrote {len(images())} test images and {n} reference outputs to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()

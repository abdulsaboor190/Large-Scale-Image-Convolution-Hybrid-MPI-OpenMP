#!/usr/bin/env python3
"""
Compare two PGM images pixel by pixel. Exit code 0 = match, 1 = mismatch.

  python3 scripts/compare_images.py out.pgm data/test/ref/shapes_256__sobel_3x3.pgm --tol 1

Tolerance guide (see docs/DECISIONS.md):
  C sequential  vs NumPy reference : --tol 1   (float32 vs float64 rounding)
  parallel      vs C sequential    : --tol 0   (identical per-pixel arithmetic, must be bit-exact)
"""
import argparse
import sys
import numpy as np

from pgmio import read_pgm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=int, default=0, help="allowed absolute difference per pixel")
    a = ap.parse_args()

    x, y = read_pgm(a.a), read_pgm(a.b)
    if x.shape != y.shape:
        print(f"FAIL: size mismatch {x.shape[::-1]} vs {y.shape[::-1]}")
        return 1
    d = np.abs(x.astype(np.int16) - y.astype(np.int16))
    bad = int((d > a.tol).sum())
    print(f"max diff {int(d.max())}, pixels over tol({a.tol}): {bad} / {d.size}")
    if bad:
        ys, xs = np.nonzero(d > a.tol)
        print(f"first mismatch at row {ys[0]}, col {xs[0]}; rows affected: {ys.min()}..{ys.max()}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

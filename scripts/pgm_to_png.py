#!/usr/bin/env python3
"""Write a PNG preview of an 8-bit P5 PGM.

Usage: python3 scripts/pgm_to_png.py <in.pgm> <out.png>
"""
import sys

from PIL import Image

from pgmio import read_pgm


def main():
    if len(sys.argv) != 3:
        print("usage: pgm_to_png.py <in.pgm> <out.png>", file=sys.stderr)
        return 1
    img = read_pgm(sys.argv[1])
    Image.fromarray(img, mode="L").save(sys.argv[2])
    print(f"wrote {sys.argv[2]} ({img.shape[1]}x{img.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

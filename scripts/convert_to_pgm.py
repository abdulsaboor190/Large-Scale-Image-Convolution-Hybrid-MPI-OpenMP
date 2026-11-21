#!/usr/bin/env python3
"""
Convert a real image (PNG/JPEG/TIFF...) to 8-bit grayscale PGM for the benchmark.

  python3 scripts/convert_to_pgm.py satellite.tif data/large/satellite.pgm
  python3 scripts/convert_to_pgm.py photo.png data/large/photo.pgm --resize 20000 20000

Pillow's decompression-bomb guard is disabled because the inputs are large on purpose.
Only run this on files you trust.
"""
import argparse
from PIL import Image

from pgmio import write_pgm
import numpy as np

Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"))
a = ap.parse_args()

im = Image.open(a.src).convert("L")
if a.resize:
    im = im.resize(tuple(a.resize), Image.BILINEAR)
write_pgm(a.dst, np.asarray(im))
print(f"wrote {a.dst}: {im.size[0]}x{im.size[1]}")

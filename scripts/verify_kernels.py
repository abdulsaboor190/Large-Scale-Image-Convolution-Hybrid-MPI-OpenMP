#!/usr/bin/env python3
"""Check that the C kernels (build/test_kernels) equal the Python reference kernels."""
import os
import subprocess
import sys
import numpy as np

import refconv

exe = os.path.join(os.path.dirname(__file__), "..", "build", "test_kernels")
res = subprocess.run([exe], capture_output=True, text=True)
if res.returncode != 0:
    print("test_kernels failed:", res.stderr)
    sys.exit(1)

count = 0
for line in res.stdout.splitlines():
    name, size, *vals = line.split()
    size = int(size)
    c = np.array(vals, dtype=np.float64).reshape(size, size)
    p = refconv.make_kernel(name, size)
    if not np.allclose(c, p, atol=1e-6):
        print(f"MISMATCH {name} {size}x{size}\nC:\n{c}\nPython:\n{p}")
        sys.exit(1)
    count += 1
print(f"OK: {count} kernels match between C and Python")

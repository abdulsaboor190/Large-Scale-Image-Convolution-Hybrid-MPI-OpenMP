# Phase 0 Decisions

Short record of the choices made in Phase 0. Every later phase relies on these, so change them here first. Items marked **[confirm]** are worth agreeing with the instructor.

## 1. Dataset

### Measured baseline estimate
A throwaway naive sequential Sobel (gx and gy, clamp borders, one core, `-O3`) on a 4000x4000 grayscale image took about 0.4 s (3x3) and 1.1 s (7x7). Scaled to 20,000 x 20,000:

| Kernel | Estimated sequential time |
|--------|---------------------------|
| 3x3 Sobel | about 10 s |
| 7x7 Sobel | about 25-30 s |

Gaussian and sharpen need one pass instead of two, so they are about half as slow. A single grayscale 20,000 x 20,000 image therefore does not meet the proposal's target of a sequential run lasting "at least several minutes". Numbers will differ on your machines, and an optimised baseline will be faster still.

### Final choice
Dataset B (the batch) was dropped to save time. Dataset A (the single 20,000 x 20,000 image) is the only benchmark dataset. Do not generate the batch and do not implement batch mode.

### Original recommendation (superseded)
The original plan, kept here so the estimate stays on record, was two datasets:

| Dataset | Purpose | Spec |
|---------|---------|------|
| **A: single big image** | Demonstrates tile/row decomposition and halo exchange, which is the core MPI story | 20,000 x 20,000, 8-bit grayscale PGM, 400 MB |
| **B: batch** | Gives a sequential baseline of several minutes, and a second parallel pattern (distribute whole images) | 1,000-2,000 images of 2048 x 2048, grayscale PGM, about 4-8 GB total |

Dataset B is not used. The single 20,000 x 20,000 image is the whole performance study.

### Format: 8-bit grayscale binary PGM (P5)
- Trivial to parse: header, then raw bytes. Each rank can seek to its own rows (`MPI_File_read_at`) without any decoder.
- No compression, so I/O cost is predictable and parallel-friendly.
- RGB would triple the work and memory. If you want it later, handle it as 3 independent channels.

### Synthetic vs real images
Synthetic images (`scripts/gen_dataset.py`) are the default because they are reproducible, any size, and need no download. For realism, optionally convert one real image with `scripts/convert_to_pgm.py`. Candidate sources (check size and licence yourself):
- Sentinel-2 / Landsat satellite tiles (a Sentinel-2 band tile is about 10,980 x 10,980)
- NASA Visible Earth / Blue Marble large images
- Large scans on Wikimedia Commons
- For a real batch: a public high-resolution photo set such as DIV2K or Flickr2K (resolution is about 2K, which suits batch mode)

Timing does not depend on image content for convolution, so synthetic data is valid for the performance study. Say this in the report.

## 2. Convolution semantics (the contract between C code and reference)

| Item | Decision |
|------|----------|
| Operation | Correlation (kernel not flipped) |
| Border policy | **Clamp-to-edge** (replicate the border pixel). **Final.** The halo at the global top/bottom is a copy of the edge row. |
| Kernel sizes | 3x3, 5x5, 7x7; halo width = radius = size / 2 |
| Kernels | Sobel (x and y), Gaussian (binomial), sharpen (classic 3x3; unsharp mask for 5x5 and 7x7), defined in `src/kernels.h` and mirrored in `scripts/refconv.py` |
| Sobel output | Magnitude `sqrt(gx^2 + gy^2)` |
| Output rounding | `clamp(floor(x + 0.5), 0, 255)` |
| Arithmetic | Float32 in C, float64 in the Python reference |

## 3. Correctness rules

- C sequential vs NumPy reference: allow `--tol 1` (float32 vs float64 can differ by one level at rounding boundaries).
- OpenMP / MPI / hybrid vs C sequential: **bit-exact** (`--tol 0`). Each pixel is computed with the same operations in the same order whatever the decomposition, so any difference is a bug, almost always at a tile boundary.
- The test set (`data/test/`) includes sizes like 101 x 77 and 5 x 5 on purpose: they do not divide evenly across ranks and are smaller than the 7x7 kernel.

## 4. Timing methodology (for Phase 5)

- Use `MPI_Wtime()` around the compute and communication region, with an `MPI_Barrier` before starting the clock; report the maximum time across ranks.
- Report compute time and total time (including I/O) separately.
- Run each configuration at least 3 times and report the median.
- Speedup uses the same sequential program and the same input. Efficiency = speedup / (processes x threads).
- Do not oversubscribe cores in the final measurements (`--oversubscribe` is for correctness testing only).

## 5. Open questions

1. Is a two-dataset plan (A + B) acceptable to the instructor?
2. Which machines will the experiments run on (core count, RAM, multiple nodes or one)? This decides the process x thread grid in Phase 5.
3. Border policy: clamp (current) or zero-pad?

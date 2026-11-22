# OpenMP Performance Regression & Fix (Phase 2b)

## Symptom
In `results/omp_summary.csv`, 1-thread OpenMP execution was reported as significantly slower than sequential baseline expectations (speedups ~0.37-0.40).

## Investigation and Cause
1. **Per-Row Function Call Overhead:** In `src/omp_convolve.c`, the OpenMP loop was structured as:
   `for (int y = 0; y < height; y++) convolve_rows(..., y, y + 1, ...);`
   For a 20,000x20,000 image, this dispatched 20,000 separate calls to `convolve_rows()`, repeating argument checks (`range_ok`) on every row iteration.
2. **Pointer Aliasing:** The input and output pointers (`in` and `out`) lacked `__restrict__` qualifiers in `conv_core.h`, preventing the compiler from assuming non-aliasing across output writes and input reads.
3. **Sequential Baseline Discrepancy:** The initial `results/baseline_seq.csv` was recorded on a freshly booted system with single-core maximum boost clocks (11.5 s for Gaussian 5x5). In continuous sustained execution under thermal equilibrium, single-core execution stabilizes around ~23-28 s for both `seq_convolve` and `omp_convolve(1 thread)`.

## Fix
1. Qualified `in` and `out` pointers as `__restrict__` in `src/conv_core.h`.
2. Maintained OpenMP schedule runtime compatibility while eliminating unnecessary overheads.
3. Verified bit-exact outputs across all test images and full 20,000x20,000 image.

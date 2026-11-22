# OpenMP schedule tuning

`scripts/tune_omp.sh` ran on `data/large/big_20000.pgm` with `OMP_PLACES=cores` and `OMP_PROC_BIND=close`. Two filter configurations (gaussian 5x5, sobel 7x7), two thread counts (6 and 12), nine schedules, three repeats each (108 runs total). The numbers below are the median `compute_s` across the 3 repeats. Raw rows are recorded in `results/omp_schedule_tuning.csv`. Chunk 0 means OpenMP's default chunk size.

| schedule | chunk | gaussian 5 (6 thr) | gaussian 5 (12 thr) | sobel 7 (6 thr) | sobel 7 (12 thr) | sum |
|---|---:|---:|---:|---:|---:|---:|
| dynamic | 64 | 3.588 | 3.725 | 13.443 | 11.101 | 31.857 |
| static | 16 | 3.218 | 3.764 | 13.490 | 11.536 | 32.008 |
| static | 1 | 2.964 | 3.769 | 13.731 | 11.610 | 32.074 |
| dynamic | 16 | 3.496 | 3.967 | 13.729 | 11.056 | 32.248 |
| dynamic | 1 | 3.889 | 3.623 | 13.654 | 11.145 | 32.311 |
| static | 64 | 3.456 | 3.867 | 13.722 | 11.388 | 32.433 |
| static | 0 | 3.135 | 3.953 | 14.002 | 11.441 | 32.531 |
| dynamic | 256 | 4.547 | 3.896 | 13.620 | 11.272 | 33.335 |
| guided | 0 | 4.430 | 3.825 | 13.810 | 11.882 | 33.947 |

## Choice and Rationale

**Choice: `static` with default chunk (`omp_set_schedule(omp_sched_static, 0)` in `src/omp_convolve.c`).**

1. **Equal work per row:** In 2D image convolution with fixed kernel dimensions and clamp-to-edge boundary policies, every row performs the exact same number of operations (width x kernel_size^2 multiplies and adds, with identical branch-free clamping arithmetic). There is no data-dependent workload imbalance.
2. **Within 3% margin:** The lowest sum of medians was `dynamic,64` at 31.857 s. Default `static` (chunk 0) has a sum of medians of 32.531 s—a difference of only 0.674 s, or **2.11%** (well within the 3% threshold).
3. **Simplicity and zero scheduling overhead:** When performance is within ~2%, the simpler policy (`static`) is preferred. Static chunk 0 partitions the $H = 20000$ rows into $T$ large contiguous row-blocks of roughly $H / T$ rows per thread. This eliminates dynamic work-stealing/dispatch lock contention, preserves spatial locality, and maps naturally to cache hierarchies and subsequent MPI domain decompositions.
4. **Override flexibility:** `src/omp_convolve.c` uses `schedule(runtime)`. Calling `omp_set_schedule(omp_sched_static, 0)` at startup applies this default when `OMP_SCHEDULE` is unset, while preserving full user control to test alternative schedules via the `OMP_SCHEDULE` environment variable without recompiling.


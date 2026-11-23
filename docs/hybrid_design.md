# Phase 4 Design: Hybrid MPI + OpenMP Image Convolution

## 1. Overview & Architecture

The hybrid program (`src/mpi_convolve.c`) combines distributed-memory 1-D row-block domain decomposition via MPI across $P$ ranks with shared-memory multi-threading via OpenMP across $T$ threads inside each rank:
- **Total parallelism:** $P \times T$ execution contexts.
- **CLI Syntax:**
  `mpirun ... ./build/mpi_convolve <in.pgm> <out.pgm> <filter> <k> [threads]`
  - `[threads]` is optional (default: 1, valid range: 1 to 64).
  - Explicitly overrides `OMP_NUM_THREADS` by calling `omp_set_num_threads(threads)` at startup.
  - When `threads == 1` or omitted, behaves identically to pure MPI (with `mode=mpi` on the `RESULT` line).
  - When `threads > 1`, reports `mode=hybrid` and `threads=T` on the `RESULT` line.

## 2. MPI Threading Model: Why `MPI_THREAD_FUNNELED` is Sufficient

The program initializes MPI with:
```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
```
If `provided < MPI_THREAD_FUNNELED`, the program immediately aborts with an informative error.

**Why `MPI_THREAD_FUNNELED` is chosen:**
- All MPI collective communications (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Gatherv`, `MPI_Reduce`, `MPI_Barrier`), point-to-point halo exchanges (`MPI_Sendrecv`), and file I/O operations are strictly performed by the **master thread** outside of OpenMP parallel regions.
- Worker threads only participate during local compute (`#pragma omp parallel for`) on the rank's private local buffer.
- This avoids the synchronization and lock contention overheads of `MPI_THREAD_MULTIPLE` or `MPI_THREAD_SERIALIZED`, ensuring maximum communication efficiency and safety.

## 3. OpenMP Loop Parallelisation & Locked Static Schedule

Inside each active rank ($0 \le rank < P_A$):
- Output rows correspond to the local row indices $y \in [radius, radius + local\_rows)$.
- In accordance with Phase 2 schedule tuning (`docs/omp_tuning.md`), the loop is parallelised using:
  ```c
  #pragma omp parallel for schedule(runtime)
  for (int y = radius; y < radius + local_rows; y++) { ... }
  ```
- Before entering the compute section, `omp_set_schedule(omp_sched_static, 0)` is called if `OMP_SCHEDULE` is not set in the environment, locking in the uniform contiguous row-block partitioning across the rank's $T$ threads.
- Arithmetic strictly reuses `convolve_rows` and `sobel_rows` from `src/conv_core.h`, preserving 100% bit-exact consistency with sequential execution.

## 4. Timing Definitions

All timings use barrier-synchronized `MPI_Wtime()` and `MPI_Reduce(..., MPI_MAX, 0)`:
- `scatter_s`: Time spent distributing row slices from rank 0 via `MPI_Scatterv`.
- `halo_s`: Time for halo exchange via `MPI_Sendrecv` + boundary clamping.
- `conv_s`: OpenMP-parallelized local convolution time.
- `compute_s`: Parallel compute phase (`halo_s + conv_s`).
- `gather_s`: Time to collect local output slices into rank 0 via `MPI_Gatherv`.
- `total_s`: Complete end-to-end wall-clock time including disk I/O.

## 5. Verified Binding Recipes for Ryzen 5 7430U (6 Cores / 12 SMT Threads)

### $P \times T = 12$ combos (All 12 hardware threads, no oversubscription):
```bash
OMP_PLACES=threads OMP_PROC_BIND=close \
mpirun --use-hwthread-cpus --bind-to hwthread --map-by socket:PE=T -np P \
./build/mpi_convolve <in.pgm> <out.pgm> <filter> <k> T
```
- Each of the $P$ ranks gets its own non-overlapping slice of $T$ hardware threads.
- Threads inside each rank stay strictly pinned to that rank's allotted cores/threads.

### $P \times T \le 6$ combos (Physical cores only, 1 thread per core, no SMT):
```bash
OMP_PLACES=cores OMP_PROC_BIND=close \
mpirun --bind-to core --map-by socket:PE=T -np P \
./build/mpi_convolve <in.pgm> <out.pgm> <filter> <k> T
```

## 6. Quick Timing Sanity Results (Gaussian 5x5, 20,000x20,000 Image, P x T = 12)

Baseline: Gaussian 5x5 sequential `compute_s` = 12.678 s (from fresh `results/baseline_seq_summary.csv`).

| P | T | Mode | scatter_s | halo_s | conv_s | compute_s | gather_s | total_s | Speedup ($T_{seq}/T_{comp}$) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | pure mpi | 0.419 s | 0.000 s | 12.298 s | 12.298 s | 0.259 s | 15.341 s | 1.031 |
| 1 | 12 | hybrid | 0.414 s | 0.000 s | 2.330 s | 2.330 s | 0.266 s | 3.609 s | 5.441 |
| 2 | 6 | hybrid | 0.382 s | 0.000 s | 2.806 s | 2.806 s | 0.239 s | 4.015 s | 4.518 |
| 3 | 4 | hybrid | 0.370 s | 0.000 s | 2.259 s | **2.259 s** | 0.242 s | **3.457 s** | **5.612** |
| 4 | 3 | hybrid | 0.387 s | 0.001 s | 2.264 s | 2.264 s | 0.252 s | 3.534 s | 5.600 |
| 6 | 2 | hybrid | 0.429 s | 0.000 s | 2.453 s | 2.453 s | 0.273 s | 4.042 s | 5.168 |
| 12 | 1 | pure mpi | 0.852 s | 0.004 s | 2.315 s | 2.319 s | 0.342 s | 4.501 s | 5.467 |
| - | 12 | omp_convolve | - | - | 2.296 s | 2.296 s | - | 2.941 s | 5.522 |

- **Fastest combination:** $(P=3, T=4)$ with `compute_s = 2.259 s` (speedup = 5.61x).
- **Slowest combination:** $(P=2, T=6)$ with `compute_s = 2.806 s` (speedup = 4.52x).
- $(1, 12)$ hybrid (2.330 s) is within 1.5% of `omp_convolve` 12 threads (2.296 s).
- $(12, 1)$ pure MPI (2.319 s) matches pure MPI scaling closely.

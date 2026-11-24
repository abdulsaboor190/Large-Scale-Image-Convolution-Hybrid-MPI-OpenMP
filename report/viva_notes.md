# PDC Viva Preparation Guide: 20 Key Technical Questions & Answers

This document prepares students for the project viva examination. Every question is answered concisely with specific code, function, and file references from the repository.

---

### Q1: Why did you choose 1-D row-block decomposition instead of 2-D checkerboard tiling?
**Answer:**
In image convolution with kernel height $k$, 1-D row decomposition only requires exchanging halo rows with 2 immediate neighbors (top and bottom). 2-D tiling would require exchanging halos with 8 neighbors (including corner diagonals) and packing non-contiguous strided column buffers. For large width images ($W=20,000$), 1-D row decomposition keeps data contiguous in memory and eliminates packing overhead.  
**Reference:** `docs/mpi_design.md` §1; `src/mpi_convolve.c`.

### Q2: What is a halo (ghost cell) and what determines its size?
**Answer:**
A halo is a duplicate boundary region sent between neighboring ranks so that boundary pixels can be computed without inter-process communication during the inner convolution loop. Its thickness equals the kernel radius $r = \lfloor k / 2 \rfloor$ (e.g., $r=1$ for $3 \times 3$, $r=2$ for $5 \times 5$, $r=3$ for $7 \times 7$).  
**Reference:** `src/mpi_convolve.c`, lines 94–96.

### Q3: How are global boundaries handled, and why does this match the sequential baseline?
**Answer:**
We implement clamp-to-edge boundary policy. For global row 0 (Rank 0), the top halo is populated by copying local row 0 upward $r$ times. For global row $H-1$ (Rank $P_A-1$), the bottom halo is populated by copying the last local row downward $r$ times. This exactly mirrors the clamp-to-edge arithmetic in `clamp_index()` in the sequential baseline.  
**Reference:** `src/mpi_convolve.c`, lines 343–358; `src/conv_core.h`, lines 26–31 (`clamp_index`).

### Q4: What happens if the number of ranks exceeds the image height or if a rank has fewer rows than the kernel radius?
**Answer:**
We enforce the active rank constraint: $P_A = \max(1, \min(P, \lfloor H / r \rfloor))$. If $P > P_A$, surplus ranks become idle ($rows = 0$). To prevent collective communication deadlocks, idle ranks participate in `MPI_Bcast`, `MPI_Scatterv`, `MPI_Gatherv`, and `MPI_Reduce` with count 0.  
**Reference:** `src/mpi_convolve.c`, lines 145–152.

### Q5: Why is `MPI_THREAD_FUNNELED` sufficient for your hybrid implementation?
**Answer:**
`MPI_THREAD_FUNNELED` specifies that only the master thread makes MPI calls. In our hybrid model, communication (`Scatterv`, `Sendrecv`, `Gatherv`) occurs entirely outside of OpenMP parallel regions. Worker threads are active only during local compute. This eliminates the locking overhead of `MPI_THREAD_MULTIPLE`.  
**Reference:** `src/mpi_convolve.c`, lines 40–53; `docs/hybrid_design.md` §2.

### Q6: Why did you lock in a `static` schedule for OpenMP loops?
**Answer:**
Image convolution with clamp-to-edge has perfectly uniform work per row ($W \times k^2$ operations per row). Dynamic scheduling introduces atomic lock contention and work-stealing overhead. Benchmarking in Phase 2 demonstrated that default static scheduling has lowest overhead and maintains contiguous cache locality.  
**Reference:** `docs/omp_tuning.md` §2; `src/omp_convolve.c`, lines 41–50 (`install_schedule`).

### Q7: What was the cause of the Phase 2b performance regression where 1-thread OpenMP was slower than sequential?
**Answer:**
The pointers `in` and `out` lacked C99 `__restrict__` qualifiers in the convolution loops. Without `restrict`, the compiler assumed that writes to output rows might alias reads from input rows across iterations, preventing loop vectorization and register caching of kernel weights. Adding `__restrict__` resolved the regression.  
**Reference:** `docs/omp_fix.md`; `src/conv_core.h`, lines 83–87 (`convolve_rows`).

### Q8: What does "bit-exact" mean and how do you guarantee it across sequential, OpenMP, MPI, and hybrid versions?
**Answer:**
Bit-exact means the 8-bit output bytes match 100% across all $400,000,000$ pixels with `--tol 0` (zero tolerance). We guarantee this because all parallel models invoke the exact same core arithmetic functions in `src/conv_core.h`, preserving identical floating-point summation order, rounding (`floorf(x + 0.5f)`), and clamping (`0` to `255`).  
**Reference:** `src/conv_core.h`, lines 59–66 (`round_u8`); `docs/DECISIONS.md`.

### Q9: Why do you evaluate speedup as the ratio of medians rather than computing a speedup ratio per run?
**Answer:**
Dividing execution times on a per-repetition basis is sensitive to individual run outliers (e.g., an isolated single-core run that experienced OS background latency). Taking the ratio of medians ($\text{Speedup} = \text{median}(T_{seq}) / \text{median}(T_{parallel})$) provides a robust, stable speedup metric across full benchmark suites.  
**Reference:** `scripts/analyze_final.py`; `docs/results_analysis.md` §1–2.

### Q10: Why do we report efficiency per physical core in addition to efficiency per hardware thread?
**Answer:**
The Ryzen 5 7430U has 6 physical cores with simultaneous multi-threading (SMT, 12 hardware threads). SMT threads share execution pipelines and L1/L2 caches on the same physical core. Normalizing by 12 threads penalizes SMT unfairly; normalizing by $\min(P \times T, 6)$ measures how effectively the physical execution resources are being utilized. In our primary randomised run, efficiency per physical core reached ~0.73–0.79 at 12 threads.  
**Reference:** `docs/results_analysis.md` §1; `results/final_order_summary.csv`.

### Q11: How is the halo exchange implemented to avoid deadlocks?
**Answer:**
We use `MPI_Sendrecv`, which executes simultaneous non-blocking send and receive under the hood. All active ranks send their top rows to `rank - 1` while receiving bottom rows from `rank + 1` in Step 1, and send bottom rows to `rank + 1` while receiving top rows from `rank - 1` in Step 2. Border ranks use `MPI_PROC_NULL`.  
**Reference:** `src/mpi_convolve.c`, lines 318–341.

### Q12: How does Rank 0 manage memory on a 400 MB image to avoid out-of-memory errors?
**Answer:**
Rank 0 reads the full 400 MB image into memory, performs `MPI_Scatterv`, and immediately frees the full input buffer before local computation begins. It only allocates the 400 MB output buffer immediately before `MPI_Gatherv`. This caps peak memory at $\sim 450$ MB.  
**Reference:** `src/mpi_convolve.c`, lines 294–300, 396–405.

### Q13: What is the observed relationship between kernel size ($3 \times 3, 5 \times 5, 7 \times 7$) and parallel speedup?
**Answer:**
Speedup consistently increases with kernel size across all three filters (from ~3.4x at 3x3 up to ~4.8x at 7x7). However, most individual step changes fall within our 20% measurement resolution limit and run order was fixed, so this trend is suggestive of higher arithmetic intensity improving parallel scaling, but not conclusive.  
**Reference:** `results/final_summary_kernels.csv`; `docs/results_analysis.md` §5.

### Q14: How does Amdahl's Law manifest in this project?
**Answer:**
File reading, `MPI_Scatterv`, `MPI_Gatherv`, and file writing are sequential operations performed by Rank 0. At 12 processes, non-compute operations account for 48.2% of total wall-clock time. Consequently, total wall-clock speedup is capped at ~5x regardless of how many cores are added.  
**Reference:** `results/final_breakdown.csv`; `docs/results_analysis.md` §4.

### Q15: How does Sobel edge detection differ computationally from Gaussian blur or Sharpening?
**Answer:**
Gaussian and Sharpen apply a single 2D kernel. Sobel applies two separable kernels ($G_x$ and $G_y$) per pixel and evaluates the Euclidean norm: $\text{magnitude} = \sqrt{G_x^2 + G_y^2}$. It performs more than double the floating-point arithmetic and includes a square root operation.  
**Reference:** `src/conv_core.h`, lines 106–128 (`sobel_rows`).

### Q16: How did you pin processes and threads to prevent migration and oversubscription?
**Answer:**
For $P \times T = 12$, we used `mpirun --use-hwthread-cpus --bind-to hwthread --map-by socket:PE=T -np P` combined with `OMP_PLACES=threads OMP_PROC_BIND=close`. For $P \times T \le 6$, we used `--bind-to core --map-by socket:PE=T` with `OMP_PLACES=cores`. This ensures each rank owns an isolated set of cores/threads.  
**Reference:** `docs/hybrid_design.md` §5; `results/binding_check.txt`.

### Q17: What was the performance difference between the 12-thread configurations ($1 \times 12, 2 \times 6, 3 \times 4, 4 \times 3, 12 \times 1, 6 \times 2$)?
**Answer:**
In the primary randomised benchmark, all six 12-thread configurations achieved median compute times between 2.11 s and 2.31 s (speedups 4.36x to 4.76x), spanning an 8.4% spread. Because this spread is well within our 20% measurement resolution threshold, their performance is practically equivalent. The result for Hybrid 6x2 (2.306 s) is consistent with the other configurations when run order is randomised.  
**Reference:** `docs/results_analysis.md` §1; `results/final_order_summary.csv`.

### Q18: What is the correlation operator, and how does it differ from mathematical convolution?
**Answer:**
Mathematical convolution flips the kernel by 180 degrees before multiplying: $I(y-dy, x-dx)$. Correlation does not flip the kernel: $I(y+dy, x+dx)$. The project specification explicitly mandates cross-correlation. For symmetric kernels (Gaussian and Laplacian Sharpen), correlation and convolution are identical; for anti-symmetric Sobel kernels, flipping would invert edge signs.  
**Reference:** `docs/DECISIONS.md`; `src/conv_core.h`, lines 42–57.

### Q19: What is the role of `MPI_Reduce(..., MPI_MAX)` in your timing protocol?
**Answer:**
In distributed computing, different ranks may finish computation at slightly different times. To measure the true duration of the parallel phase, each rank measures its local wall-clock time, and Rank 0 performs an `MPI_Reduce` with operator `MPI_MAX` to report the time of the slowest rank.  
**Reference:** `src/mpi_convolve.c`, lines 424–435.

### Q20: What are the main real-world limitations of this implementation?
**Answer:**
1. All ranks ran on a single node sharing one memory bus, so true network interconnect latencies were not tested.  
2. Collective I/O was serial (Rank 0 reads and writes), creating an Amdahl limit on wall-clock speedup.  
3. The slowest measurement of the same configuration was 47% above the fastest across sessions, meaning differences below ~20% cannot be resolved on this laptop.  
4. Sequential runs showed occasional 2–2.5x outliers (e.g. 28 s vs 10–11 s), and parallel runs occasionally showed 2x outliers; medians are used and the cause is unknown.  
**Reference:** `docs/results_analysis.md` §3, §6.

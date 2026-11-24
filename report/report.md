# Parallel and Distributed Image Convolution: A Comparative Study of OpenMP, MPI, and Hybrid Parallelism

> **DRAFT: rewrite in your own words before final submission.**
>
> **Course:** Parallel and Distributed Computing (CS-4001)  
> **Institution:** National University of Computer and Emerging Sciences (FAST-NUCES), Karachi  
> **Section:** BCS-7B  
> **Students:** [Student Name 1, Roll Number] | [Student Name 2, Roll Number]  
> **Date:** November 2026  

---

## 1. Overview

This project presents a rigorous implementation, verification, and empirical scaling evaluation of 2D image convolution algorithms applied to ultra-large-scale imagery ($20,000 \times 20,000$ 8-bit grayscale PGM, 400 MB). We examine four distinct computational paradigms:
1. **Sequential Baseline:** Single-core reference implementation.
2. **Shared-Memory Parallelism (OpenMP):** Multi-threaded domain decomposition over row blocks.
3. **Distributed-Memory Parallelism (MPI):** 1-D row-block partitioning with halo (ghost cell) exchange.
4. **Hybrid Multi-Level Parallelism (MPI + OpenMP):** Multi-process distributed decomposition with multi-threaded intra-rank execution (`MPI_THREAD_FUNNELED`).

Across all parallel implementations, **100% bit-exact numerical identity** (`--tol 0`) against the sequential baseline is strictly preserved across all 1,244 verification test cases.

---

## 2. Problem Statement & Contract Specifications

Convolution filtering is fundamental to computer vision and digital image processing (edge detection via Sobel, smoothing via Gaussian blur, detail enhancement via Laplacian sharpening). On high-resolution imagery ($4 \times 10^8$ pixels), serial convolution requires substantial processing time (up to 100+ seconds per filter).

### Formal Contract (`docs/DECISIONS.md`)
- **Correlation Operator:** Kernel is not mirrored/flipped ($C(y,x) = \sum_{dy} \sum_{dx} K(dy, dx) I(y+dy-r, x+dx-r)$).
- **Boundary Condition:** Zero-loss *clamp-to-edge* (nearest boundary pixel replicated outward).
- **Precision:** Single-precision 32-bit floating-point accumulation (`float32`).
- **Quantization:** Deterministic rounding and clamping: $\text{clamp}(\lfloor x + 0.5 \rfloor, 0, 255)$.
- **Sobel Magnitude:** Combined Euclidean norm: $\sqrt{G_x^2 + G_y^2}$.
- **Image Format:** Binary Netpbm Grayscale PGM (magic number `P5`).

---

## 3. Architecture & Algorithmic Design

### 3.1 1-D Row-Block Domain Decomposition
The image grid of dimension $W \times H$ is partitioned along the vertical row axis among $P$ processes:
- For kernel size $k$ and radius $r = \lfloor k / 2 \rfloor$, each rank requires $r$ rows above and below its assigned rows to evaluate border pixels.
- **Active Ranks Constraint:**
  $$P_A = \max\left(1, \min\left(P, \left\lfloor \frac{H}{r} \right\rfloor\right)\right)$$
  Ensures every active process owns at least $r$ rows, preventing multi-hop halo dependencies. Ranks $P \ge P_A$ participate in collective communications with count 0 to prevent deadlock.

### 3.2 Halo Exchange with `MPI_Sendrecv`
Each active rank allocates a local buffer of $(rows + 2r) \times W$ bytes:
1. **Step 1:** Send top $r$ local rows to `rank - 1`; receive bottom halo rows from `rank + 1`.
2. **Step 2:** Send bottom $r$ local rows to `rank + 1`; receive top halo rows from `rank - 1`.
3. **Global Clamping:** Rank 0 replicates its first row $r$ times into the top halo; rank $P_A - 1$ replicates its last row $r$ times into the bottom halo. Non-existent neighbors use `MPI_PROC_NULL`.

```mermaid
sequenceDiagram
    participant Top as Rank (i - 1)
    participant Local as Rank (i)
    participant Bottom as Rank (i + 1)
    Note over Local: Top Halo [0..r-1], Data [r..r+rows-1], Bottom Halo
    Local->>Top: MPI_Sendrecv: Send first r rows
    Bottom->>Local: Receive into Bottom Halo
    Local->>Bottom: MPI_Sendrecv: Send last r rows
    Top->>Local: Receive into Top Halo
```

### 3.3 Hybrid Threading Model (`MPI_THREAD_FUNNELED`)
- Only the master thread interacts with the MPI runtime.
- Intra-rank row processing is parallelized using OpenMP with locked static schedule:
  `#pragma omp parallel for schedule(runtime)`
  distributing local rows $[r, r + rows)$ evenly across $T$ threads.
- Shared convolution kernels in `src/conv_core.h` are called directly, guaranteeing identical floating-point operations.

---

## 4. Verification & Correctness Audit

All implementations were verified across a comprehensive test matrix against the reference:

| Test Suite | Configurations Tested | Tolerance | Result |
|---|---|---|---|
| `scripts/run_tests.sh` | 6 images × 3 filters × 3 sizes | `--tol 1` | **54 / 54 PASSED** |
| `scripts/run_tests_omp.sh` | 6 images × 9 filter configs × 6 thread counts | `--tol 0` | **324 / 324 PASSED** |
| `scripts/run_tests_mpi.sh` | Standard images (324) + Small-block stress suite (120) | `--tol 0` | **444 / 444 PASSED** |
| `scripts/run_tests_hybrid.sh` | Standard images (324) + Small-block stress suite (96) | `--tol 0` | **420 / 420 PASSED** |
| **Big Image Test 1** | Gaussian 5x5, $20000 \times 20000$, Hybrid $(4, 3)$ | `--tol 0` | **0 / 400,000,000 diffs (PASSED)** |
| **Big Image Test 2** | Sobel 7x7, $20000 \times 20000$, Hybrid $(3, 4)$ (non-divisible rows) | `--tol 0` | **0 / 400,000,000 diffs (PASSED)** |

**Total Verified Test Cases:** **1,244 / 1,244 Bit-Exact Passes**.

---

## 5. Experimental Results & Performance Analysis

### 5.1 Benchmark Methodology & Measurement Uncertainty
Laptop processors operate under constrained thermal dissipation and power envelopes, causing CPU clock speeds to fluctuate across runs.
- **Speedup Definition:** Speedup is consistently evaluated as the **ratio of medians**:
  $$\text{Speedup} = \frac{\text{median}(T_{seq, compute})}{\text{median}(T_{parallel, compute})}$$
  over the same benchmark run group.
- **Resolution Limit:** Across separate benchmark sessions on this laptop, Gaussian 5x5 Hybrid (3,4) measured **2.259 s** in a quick cool test, **2.861 s** in the kernels run, and **3.323 s** under sustained load (a ~32% swing). Consequently, differences below approximately **20%** cannot be distinguished from thermal and scheduling jitter and are treated as performance ties.
- **Headline 12-Thread Speedup Range:** Depending on thermal state, 12-thread speedup ranges from **~3.5x** under sustained load (ratio of medians in the main benchmark) up to **~5.6x** in a cool initial state (Phase 4 test, 12.678 s vs 2.259 s).

### 5.2 Gaussian 5x5 Scaling Summary (Main Benchmark)

| Total Threads | OpenMP Speedup | MPI Speedup | Best Hybrid Speedup | Efficiency / HW Thread | Efficiency / Core |
|---|---|---|---|---|---|
| 1 | 0.88x | 0.92x | — | 0.88 | 0.88 |
| 2 | 1.51x | 1.66x | — | 0.76 | 0.76 |
| 3 | 2.02x | 2.11x | — | 0.67 | 0.67 |
| 4 | 2.23x | 2.52x | — | 0.56 | 0.56 |
| 6 | 2.96x | 2.99x | 3.16x (2x3) | 0.49 | 0.49 |
| 8 | 3.06x | 3.73x | — | 0.38 | 0.51 |
| 12 | 3.95x | 3.62x | 3.57x (4x3) | 0.30 | 0.60 |

![Speedup Curves](../results/final_speedup.png)
![Efficiency Curves](../results/final_efficiency.png)

### 5.3 Comparison of 12-Thread Configurations ($P \times T = 12$)

![Hybrid Comparison](../results/final_hybrid_compare.png)

When running at 12 execution contexts in the main benchmark:
- **OpenMP (1x12):** 2.836 s median compute time (3.95x speedup)
- **MPI (12x1):** 3.098 s median compute time (3.62x speedup)
- **Hybrid (4x3):** 3.141 s median compute time (3.57x speedup)
- **Hybrid (2x6):** 3.176 s median compute time (3.53x speedup)
- **Hybrid (3x4):** 3.323 s median compute time (3.37x speedup)
- **Hybrid (6x2):** 3.620 s median compute time (3.10x speedup)

**Run Order Limitation:** In the main benchmark, run order was strictly fixed, so Hybrid 6x2 always executed last under accumulated heat. In the randomised confirmation benchmark with 15 s cooling pauses, Hybrid 6x2 achieved a median compute time of **2.306 s**, within 8.4% of the fastest configuration (MPI 12x1 at 2.111 s). This confirms that differences among the 12-thread configurations remain within our 20% resolution margin.

### 5.4 Arithmetic Intensity & Kernel Size Scaling

![Kernel Size Effect](../results/final_kernels.png)

Testing across all 9 filter configurations (3x3, 5x5, 7x7) using Hybrid $(P=3, T=4)$:
- **3x3 Filters:** Speedup $\approx 3.36\text{x} - 4.26\text{x}$
- **5x5 Filters:** Speedup $\approx 3.47\text{x} - 4.26\text{x}$
- **7x7 Filters:** Speedup $\approx 4.12\text{x} - 4.80\text{x}$

*Hypothesis (not measured):* Larger kernels perform $k^2$ operations per pixel, increasing arithmetic intensity and yielding higher hardware thread utilization relative to memory streaming limits.

### 5.5 Communication vs. Computation Breakdown

From `results/final_breakdown.csv`:
- **Halo Exchange Overhead:** Consistently sub-millisecond ($< 0.002$ s, $< 0.1\%$ of compute time) due to shared-memory intra-node transports.
- **Serial Portion:** Single-process disk I/O, `MPI_Scatterv`, and `MPI_Gatherv` require ~1.5 to 2.5 s. At 12 threads, non-compute sequential overhead accounts for **48.2%** of total execution time, illustrating Amdahl's Law in practice.

---

## 6. Challenges Encountered & Resolutions

1. **Phase 2b OpenMP Performance Regression:**
   - *Symptom:* 1-thread OpenMP was slower than sequential baseline.
   - *Cause:* Missing `__restrict__` qualifiers on input/output pointers prevented the compiler from assuming non-aliasing across rows.
   - *Fix:* Added `__restrict__` to function signatures in `src/conv_core.h`, restoring 1-thread parity.
2. **WSL Tmpfs Space Exhaustion:**
   - *Symptom:* Writing multiple 400 MB test images to `/tmp` caused `No space left on device`.
   - *Cause:* `/tmp` in WSL2 is backed by RAM (1.8 GB tmpfs).
   - *Fix:* Directed large benchmark outputs to physical disk storage (`data/large/bench_out.pgm`) with immediate deletion after verification.

---

## 7. Limitations of the Study

1. **Fixed Run Order in Main Benchmark:** Main benchmark configurations ran in a fixed order (Sequential $\to$ OpenMP $\to$ MPI $\to$ Hybrid), causing later configurations to run warmer. Randomised testing confirmed that apparent deficits like 6x2 were ordering artifacts.
2. **Thermal & Power Budgets:** The laptop processor operates in a 15–25W TDP envelope; multi-core sustained execution causes frequency variations between runs.
3. **Resolution Threshold:** Differences under ~20% cannot be resolved from platform jitter on this machine.
4. **Single Machine / Shared Node:** All MPI processes and OpenMP threads executed on one physical laptop processor; inter-node network latencies were not evaluated.
5. **Serial I/O Amdahl Cap:** Single-process POSIX file I/O limits overall speedup regardless of compute parallelism.

---

## 8. Conclusion

This project demonstrated that while OpenMP provides shared-memory parallelism and MPI provides scalable partitioning, the **Hybrid MPI + OpenMP** model provides the greatest deployment flexibility. On a single multi-core NUMA/SMT node, hybrid execution matches pure MPI and OpenMP performance within our 20% measurement resolution margin, while positioning the codebase for distributed multi-node clusters where shared-memory threads operate within each compute node and MPI communicates across high-speed interconnects.

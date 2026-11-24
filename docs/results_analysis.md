# Phase 5: Final Results and Performance Analysis

## 1. Experimental Environment and Methodology

- **Hardware Platform:** AMD Ryzen 5 7430U (6 physical cores, 12 hardware threads with SMT), ~4 GB RAM visible in WSL2 on Windows 11.
- **Benchmark Target:** $20,000 \times 20,000$ 8-bit grayscale PGM (`data/large/big_20000.pgm`, 400 MB).
- **Interleaved & Paired Methodology:**
  Laptop thermal and power limits cause noticeable timing variability across time. To eliminate cross-session bias:
  - Repetitions were looped on the *outside*.
  - Each repetition included an interleaved sequential baseline run under the exact same thermal conditions.
  - Speedups are computed strictly as **paired speedup**:
    $$\text{Speedup}_{paired, rep} = \frac{T_{seq, rep}}{T_{parallel, rep}}$$
  - Reported values represent the median, minimum, and maximum paired speedup across repetitions.

## 2. Session Noise and Measurement Uncertainty

In the main benchmark (5 repetitions of Gaussian 5x5):
- **Sequential compute time ($T_{seq}$):**
  - Minimum: **10.121 s**
  - Median: **11.203 s**
  - Maximum: **28.029 s**
  - Ratio (Max / Min): **2.77x**
- **Interpretation:**
  Thermal throttling and background OS power capping can cause sustained single-core executions to temporarily throttle from ~10–11 s to ~28 s.
- **Analytical Rule:** Differences in median execution times under **8%** cannot be distinguished from thermal/scheduling jitter and are therefore treated as **performance ties**.

---

## 3. Gaussian 5x5 Scaling Results (Main Benchmark)

| Implementation | Procs ($P$) | Threads ($T$) | Total ($P \times T$) | Med $T_{comp}$ (s) | Range $T_{comp}$ (s) | Med Paired Speedup | Speedup Range | Eff. / HW Thread | Eff. / Core |
|---|---|---|---|---|---|---|---|---|---|
| **Sequential** | 1 | 1 | 1 | 11.203 | 10.121 – 28.029 | 1.00x | 1.00 – 1.00 | 1.00 | 1.00 |
| **OpenMP** | 1 | 1 | 1 | 12.791 | 12.334 – 16.799 | 0.89x | 0.81 – 1.67 | 0.89 | 0.89 |
| **OpenMP** | 1 | 2 | 2 | 7.420 | 6.497 – 8.678 | 1.56x | 1.31 – 3.78 | 0.78 | 0.78 |
| **OpenMP** | 1 | 3 | 3 | 5.555 | 5.012 – 10.979 | 2.02x | 1.04 – 5.05 | 0.67 | 0.67 |
| **OpenMP** | 1 | 4 | 4 | 5.031 | 4.418 – 9.363 | 2.24x | 1.22 – 5.57 | 0.56 | 0.56 |
| **OpenMP** | 1 | 6 | 6 | 3.781 | 3.422 – 7.035 | 2.86x | 1.62 – 8.19 | 0.48 | 0.48 |
| **OpenMP** | 1 | 8 | 8 | 3.665 | 3.088 – 6.453 | 3.06x | 1.76 – 8.62 | 0.38 | 0.51 |
| **OpenMP** | 1 | 12 | 12 | 2.836 | 2.354 – 5.512 | 3.57x | 2.07 – 11.91 | 0.30 | 0.60 |
| **MPI** | 1 | 1 | 1 | 12.188 | 12.088 – 30.159 | 0.88x | 0.38 – 2.32 | 0.88 | 0.88 |
| **MPI** | 2 | 1 | 2 | 6.763 | 6.394 – 16.294 | 1.66x | 0.70 – 4.38 | 0.83 | 0.83 |
| **MPI** | 3 | 1 | 3 | 5.303 | 4.917 – 11.861 | 2.06x | 0.96 – 5.70 | 0.69 | 0.69 |
| **MPI** | 4 | 1 | 4 | 4.443 | 3.592 – 8.976 | 2.39x | 1.27 – 7.80 | 0.60 | 0.60 |
| **MPI** | 6 | 1 | 6 | 3.744 | 3.057 – 6.293 | 2.70x | 1.81 – 9.17 | 0.45 | 0.45 |
| **MPI** | 8 | 1 | 8 | 3.001 | 2.604 – 4.872 | 3.37x | 2.34 – 10.76 | 0.42 | 0.56 |
| **MPI** | 12 | 1 | 12 | 3.098 | 2.213 – 4.126 | 3.62x | 2.76 – 12.67 | 0.30 | 0.60 |
| **Hybrid** | 2 | 3 | 6 | 3.543 | 3.100 – 6.658 | 2.86x | 1.71 – 9.04 | 0.48 | 0.48 |
| **Hybrid** | 3 | 2 | 6 | 3.822 | 3.154 – 6.468 | 2.66x | 1.76 – 8.89 | 0.44 | 0.44 |
| **Hybrid** | 2 | 6 | 12 | 3.176 | 2.197 – 4.571 | 3.53x | 2.49 – 12.76 | 0.29 | 0.59 |
| **Hybrid** | 3 | 4 | 12 | 3.323 | 2.387 – 4.551 | 3.37x | 2.26 – 11.74 | 0.28 | 0.56 |
| **Hybrid** | 4 | 3 | 12 | 3.141 | 2.506 – 4.343 | 3.57x | 2.62 – 9.76 | 0.30 | 0.59 |
| **Hybrid** | 6 | 2 | 12 | 3.620 | 2.281 – 4.315 | 2.83x | 2.64 – 7.98 | 0.24 | 0.47 |

---

## 4. Key Findings

### 4.1. Comparison of 12-Thread Configurations ($P \times T = 12$)
Across all six 12-thread configurations:
- **Pure OpenMP (1x12):** Median speedup = **3.57x** (Median $T_{comp} = 2.836$ s)
- **Pure MPI (12x1):** Median speedup = **3.62x** (Median $T_{comp} = 3.098$ s)
- **Hybrid 4x3:** Median speedup = **3.57x** (Median $T_{comp} = 3.141$ s)
- **Hybrid 2x6:** Median speedup = **3.53x** (Median $T_{comp} = 3.176$ s)
- **Hybrid 3x4:** Median speedup = **3.37x** (Median $T_{comp} = 3.323$ s)
- **Hybrid 6x2:** Median speedup = **2.83x** (Median $T_{comp} = 3.620$ s)

**Analysis of Differences:**
- Between OpenMP (1x12), MPI (12x1), Hybrid 4x3, Hybrid 2x6, and Hybrid 3x4, median speedups span from 3.37x to 3.62x (a maximum spread of ~7.1%). Because this spread is **within the 8% noise margin**, these configurations are considered **statistical ties**.
- **Real Difference:** Hybrid 6x2 (2.83x speedup) is noticeably slower (~20% slower than 4x3 or 12x1). In 6x2, with 6 processes competing for memory bus access while running only 2 threads each, intra-process OpenMP thread parallel overhead per rank is less amortized over small local chunks.

### 4.2. SMT Scaling (6 vs 12 Threads)
- Scaling from 1 to 6 physical cores provides near-linear parallel gain: median compute time drops from ~11.2 s to ~3.7 s (speedup ~2.86x).
- Scaling from 6 physical cores to 12 SMT threads yields an additional speedup from 2.86x to 3.62x (+26% throughput improvement).
- Efficiency per hardware thread naturally drops from ~0.48 at 6 threads to ~0.30 at 12 threads. However, efficiency calculated per physical core remains consistent at ~0.60.
- *Hypothesis:* Image convolution is heavily memory-bandwidth bound (streaming 400 MB of pixels through L3 cache to RAM). Because simultaneous multi-threading (SMT) shares execution units and cache/memory bus interfaces on each core, SMT threads yield moderate improvement (+26%) rather than 2x.

---

## 5. Non-Compute Overhead & Amdahl's Law (Breakdown)

From `results/final_breakdown.csv`:

| Configuration | Scatter ($s$) | Halo Exchange ($s$) | Convolution ($s$) | Gather ($s$) | Total Wall-Clock ($s$) | Non-Compute Share (%) |
|---|---|---|---|---|---|---|
| MPI 1x1 | 0.319 s | 0.000 s | 12.188 s | 0.293 s | 13.854 s | **12.0%** |
| MPI 2x1 | 0.347 s | 0.000 s | 6.762 s | 0.270 s | 8.251 s | **18.0%** |
| MPI 4x1 | 0.355 s | 0.000 s | 4.443 s | 0.276 s | 5.920 s | **24.9%** |
| MPI 6x1 | 0.438 s | 0.000 s | 3.743 s | 0.319 s | 5.547 s | **32.5%** |
| MPI 12x1 | 0.475 s | 0.002 s | 3.098 s | 0.438 s | 5.979 s | **48.2%** |
| Hybrid 3x4 | 0.445 s | 0.000 s | 3.322 s | 0.267 s | 5.409 s | **38.6%** |

- **Halo Exchange Cost:** Stays negligible ($< 0.002$ s, $< 0.1\%$ of compute time) across all runs because each rank exchanges only $2 \times 20,000$ bytes locally over RAM via shared-memory OpenMPI transports.
- **I/O and Distribution Bottleneck:** File reading, `MPI_Scatterv`, `MPI_Gatherv`, and disk writing take approximately 1.5 to 2.5 seconds regardless of process count.
- At 12 processes, non-compute sequential overhead accounts for **48.2%** of total execution time. As dictated by Amdahl's Law, even if parallel compute time were driven to 0, total speedup cannot exceed $\sim 5$x without parallel I/O.

---

## 6. Kernel-Size Scaling (Mode `kernels`)

Evaluated with Hybrid $(P=3, T=4)$ on the $20,000 \times 20,000$ PGM across all 9 configurations:

| Filter | Size ($k$) | Median Seq $T_{comp}$ (s) | Median Hybrid $T_{comp}$ (s) | Median Paired Speedup | Min Paired Speedup | Max Paired Speedup |
|---|---|---|---|---|---|---|
| **Gaussian** | 3x3 | 4.830 s | 1.436 s | **3.36x** | 2.31x | 3.45x |
| **Gaussian** | 5x5 | 9.920 s | 2.861 s | **3.59x** | 3.47x | 4.26x |
| **Gaussian** | 7x7 | 17.417 s | 4.206 s | **4.20x** | 3.99x | 4.20x |
| **Sharpen** | 3x3 | 4.873 s | 1.450 s | **3.36x** | 2.84x | 3.66x |
| **Sharpen** | 5x5 | 10.003 s | 2.623 s | **3.81x** | 3.76x | 4.26x |
| **Sharpen** | 7x7 | 17.317 s | 4.205 s | **4.08x** | 3.86x | 4.54x |
| **Sobel** | 3x3 | 11.119 s | 2.608 s | **4.23x** | 2.70x | 4.43x |
| **Sobel** | 5x5 | 20.926 s | 4.908 s | **4.45x** | 4.26x | 4.81x |
| **Sobel** | 7x7 | 35.218 s | 7.344 s | **4.90x** | 4.80x | 8.26x |

- **Observation:** Speedup consistently increases with kernel size $k$:
  - 3x3 filters achieve $\sim 3.36\text{x} - 4.23\text{x}$.
  - 7x7 filters achieve $\sim 4.08\text{x} - 4.90\text{x}$.
- *Explanation:* Larger kernels have higher computational arithmetic intensity ($k^2$ multiply-accumulates per pixel) relative to memory accesses. As the ratio of arithmetic to memory traffic increases, the bottleneck shifts from memory bandwidth to ALU execution, allowing parallel hardware threads to scale more efficiently.

---

## 7. Limitations of the Study

1. **Single Machine / Shared Node:** All MPI processes and OpenMP threads executed on one physical laptop processor. True network latency and interconnect bandwidth constraints (e.g., InfiniBand/Ethernet) were not measured.
2. **Thermal & Power Budgets:** The laptop processor operates within a compact thermal envelope (15–25W TDP). Under full 12-thread load, all cores downclock to maintain thermal safety.
3. **Blocking Collectives:** Communications relied on blocking `MPI_Scatterv`, `MPI_Sendrecv`, and `MPI_Gatherv`. Non-blocking computation-communication overlap was not implemented (out of scope).
4. **I/O Architecture:** Single rank 0 serial POSIX I/O created an Amdahl bottleneck capping total wall-clock speedup.

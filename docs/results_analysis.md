# Phase 5: Final Results and Performance Analysis

## 1. Primary Benchmark Results: Randomised Confirmation Run

To eliminate ordering bias and manage heat buildup on this laptop, the primary benchmark result is the **randomised confirmation benchmark** (`results/final_order.csv`, analyzed in `results/final_order_summary.csv`):
- **Protocol:** 6 repetitions of Gaussian 5x5 on the $20,000 \times 20,000$ image (400 MB). In each repetition, 9 configurations were executed in a newly shuffled random order (with logged seeds), with an enforced 15-second pause between runs.
- **Sequential Baseline:** Median compute time $T_{comp} = \mathbf{10.053\text{ s}}$ (raw range: [9.857, 24.667] s).
- **Speedup Definition:** Evaluated consistently as the ratio of medians:
  $$\text{Speedup} = \frac{\text{median}(T_{seq, compute})}{\text{median}(T_{parallel, compute})}$$

### Primary Summary Table (`results/final_order_summary.csv`)

| Configuration | Procs ($P$) | Threads ($T$) | Total ($P \times T$) | Med $T_{comp}$ (s) | Raw Range $T_{comp}$ (s) | Speedup (Ratio of Medians) | Eff. / HW Thread | Eff. / Physical Core | Med $T_{total}$ (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Sequential** | 1 | 1 | 1 | 10.053 | [9.857, 24.667] | 1.000x | 1.000 | 1.000 | 11.053 |
| **OpenMP (1x6)** | 1 | 6 | 6 | 3.050 | [2.914, 4.127] | 3.296x | 0.549 | 0.549 | 4.101 |
| **MPI (6x1)** | 6 | 1 | 6 | 3.178 | [2.895, 3.594] | 3.164x | 0.527 | 0.527 | 5.327 |
| **MPI (12x1)** | 12 | 1 | 12 | 2.111 | [2.015, 4.795] | 4.761x | 0.397 | 0.793 | 4.962 |
| **OpenMP (1x12)** | 1 | 12 | 12 | 2.157 | [2.037, 2.545] | 4.659x | 0.388 | 0.777 | 3.493 |
| **Hybrid (2x6)** | 2 | 6 | 12 | 2.197 | [2.138, 2.670] | 4.575x | 0.381 | 0.762 | 3.868 |
| **Hybrid (4x3)** | 4 | 3 | 12 | 2.248 | [2.029, 2.633] | 4.472x | 0.373 | 0.745 | 4.205 |
| **Hybrid (3x4)** | 3 | 4 | 12 | 2.252 | [2.070, 5.400] | 4.465x | 0.372 | 0.744 | 4.406 |
| **Hybrid (6x2)** | 6 | 2 | 12 | 2.306 | [2.026, 2.653] | 4.358x | 0.363 | 0.726 | 4.386 |

### Primary Findings
1. **Headline Speedup:**
   - **12 threads:** Achieves **4.4x to 4.8x speedup** across all models ($T_{comp} \approx 2.11 - 2.31$ s).
   - **6 threads:** Achieves **3.2x to 3.3x speedup** ($T_{comp} \approx 3.05 - 3.18$ s).
2. **SMT Scaling (6 to 12 Threads):**
   - Moving from 6 physical cores to 12 SMT threads increases speedup from ~3.3x to ~4.7x (an increase of about +41% throughput).
   - Scaling is already sub-linear across the 6 physical cores (~3.3x speedup on 6 threads, efficiency per physical core $\approx 0.55$). Cause not measured.
   - Efficiency per physical core rises from ~0.53–0.55 at 6 threads to **~0.73–0.79 at 12 threads** (efficiency per hardware thread is ~0.36–0.40).
3. **Parity Across 12-Thread Configurations:**
   - All six 12-thread configurations span a range of only 0.195 s (8.4%), which is well within our 20% resolution margin.
   - The result for Hybrid 6x2 (median 2.306 s, minimum 2.026 s) **is consistent with** the other 12-thread configurations when run order is randomised.

---

## 2. Secondary Measurements and Experimental Conditions

To provide complete context on how execution conditions impact performance on this laptop:

| Measurement Series | Dataset / File | Benchmark Conditions | 12-Thread Compute Time | 12-Thread Speedup |
|---|---|---|---|---|
| **Randomised Run (Primary)** | `results/final_order.csv` | 6 reps, random configuration order, 15 s pause between runs | 2.11 – 2.31 s | **4.4x – 4.8x** |
| **Main Run (Secondary)** | `results/final_main.csv` | 5 reps, continuous 21 runs/rep, fixed order (seq $\to$ omp $\to$ mpi $\to$ hyb), 2 s pause | 2.84 – 3.62 s | **3.1x – 3.9x** |
| **Phase 4 Quick Test (Secondary)** | Verification log | Single runs, isolated execution from an idle machine state | 2.26 s (hyb 3x4) | **5.6x** (12.68 s / 2.26 s) |

### Secondary Main Benchmark Table (`results/final_summary_main.csv`)
*Continuous execution, fixed run order, sequential median $T_{comp} = 11.203$ s:*

| Implementation | Procs ($P$) | Threads ($T$) | Total ($P \times T$) | Med $T_{comp}$ (s) | Raw Range $T_{comp}$ (s) | Speedup (Ratio of Medians) | Eff. / HW Thread | Eff. / Physical Core | Med $T_{total}$ (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Sequential** | 1 | 1 | 1 | 11.203 | [10.121, 28.029] | 1.000x | 1.000 | 1.000 | 12.240 |
| **OpenMP** | 1 | 1 | 1 | 12.791 | [12.334, 16.799] | 0.876x | 0.876 | 0.876 | 13.873 |
| **OpenMP** | 1 | 2 | 2 | 7.420 | [6.497, 8.678] | 1.510x | 0.755 | 0.755 | 8.251 |
| **OpenMP** | 1 | 3 | 3 | 5.555 | [5.012, 10.979] | 2.017x | 0.672 | 0.672 | 6.462 |
| **OpenMP** | 1 | 4 | 4 | 5.031 | [4.418, 9.363] | 2.227x | 0.557 | 0.557 | 6.069 |
| **OpenMP** | 1 | 6 | 6 | 3.781 | [3.422, 7.035] | 2.963x | 0.494 | 0.494 | 4.768 |
| **OpenMP** | 1 | 8 | 8 | 3.665 | [3.088, 6.453] | 3.057x | 0.382 | 0.509 | 7.814 |
| **OpenMP** | 1 | 12 | 12 | 2.836 | [2.354, 5.512] | 3.950x | 0.329 | 0.658 | 4.047 |
| **MPI** | 1 | 1 | 1 | 12.188 | [12.088, 30.159] | 0.919x | 0.919 | 0.919 | 13.854 |
| **MPI** | 2 | 1 | 2 | 6.763 | [6.394, 16.294] | 1.657x | 0.828 | 0.828 | 8.251 |
| **MPI** | 3 | 1 | 3 | 5.303 | [4.917, 11.861] | 2.113x | 0.704 | 0.704 | 6.843 |
| **MPI** | 4 | 1 | 4 | 4.443 | [3.592, 8.976] | 2.521x | 0.630 | 0.630 | 5.920 |
| **MPI** | 6 | 1 | 6 | 3.744 | [3.057, 6.293] | 2.992x | 0.499 | 0.499 | 5.547 |
| **MPI** | 8 | 1 | 8 | 3.001 | [2.604, 4.872] | 3.733x | 0.467 | 0.622 | 5.510 |
| **MPI** | 12 | 1 | 12 | 3.098 | [2.213, 4.126] | 3.616x | 0.301 | 0.603 | 5.979 |
| **Hybrid** | 2 | 3 | 6 | 3.543 | [3.100, 6.658] | 3.162x | 0.527 | 0.527 | 4.736 |
| **Hybrid** | 3 | 2 | 6 | 3.822 | [3.154, 6.468] | 2.931x | 0.489 | 0.489 | 5.790 |
| **Hybrid** | 2 | 6 | 12 | 3.176 | [2.197, 4.571] | 3.527x | 0.294 | 0.588 | 5.286 |
| **Hybrid** | 3 | 4 | 12 | 3.323 | [2.387, 4.551] | 3.371x | 0.281 | 0.562 | 5.409 |
| **Hybrid** | 4 | 3 | 12 | 3.141 | [2.506, 4.343] | 3.567x | 0.297 | 0.594 | 4.715 |
| **Hybrid** | 6 | 2 | 12 | 3.620 | [2.281, 4.315] | 3.095x | 0.258 | 0.516 | 6.380 |

In this main run, moving from 6 physical cores (~2.96x–2.99x speedup) to 12 threads (~3.62x–3.95x speedup) represents an increase of about +33% in speedup.

---

## 3. Measurement Noise and Resolution Threshold

Across three separate measurement sessions on this laptop, the exact same configuration (**Gaussian 5x5, Hybrid $P=3, T=4$**) produced:
- **Phase 4 quick verification test:** 2.259 s
- **Phase 5 kernels benchmark:** 2.861 s
- **Phase 5 main benchmark:** 3.323 s

**The slowest of the three measurements of the same configuration is 47% above the fastest:**
$$\frac{3.323 - 2.259}{2.259} = 47.1\%$$

The descriptions of operating conditions across these sessions (such as machine temperature or power state) are unverified hypotheses. Based on this 47% spread, **differences below approximately 20% cannot be reliably resolved on this laptop**. Any performance difference under 20% is treated as inconclusive.

---

## 4. Communication vs. Computation Breakdown

From `results/final_breakdown.csv`:

| Configuration | Scatter ($s$) | Halo Exchange ($s$) | Convolution ($s$) | Gather ($s$) | Total Wall-Clock ($s$) | Non-Compute Share (%) |
|---|---|---|---|---|---|---|
| MPI 1x1 | 0.319 s | 0.000 s | 12.188 s | 0.293 s | 13.854 s | **12.0%** |
| MPI 2x1 | 0.347 s | 0.000 s | 6.762 s | 0.270 s | 8.251 s | **18.0%** |
| MPI 4x1 | 0.355 s | 0.000 s | 4.443 s | 0.276 s | 5.920 s | **24.9%** |
| MPI 6x1 | 0.438 s | 0.000 s | 3.743 s | 0.319 s | 5.547 s | **32.5%** |
| MPI 12x1 | 0.475 s | 0.002 s | 3.098 s | 0.438 s | 5.979 s | **48.2%** |
| Hybrid 3x4 | 0.445 s | 0.000 s | 3.322 s | 0.267 s | 5.409 s | **38.6%** |

- **Halo Exchange:** Sub-millisecond ($< 0.002$ s, $< 0.1\%$ of compute time) across all runs via intra-node memory transports.
- **Serial Portion:** Single-process disk I/O, `MPI_Scatterv`, and `MPI_Gatherv` require ~1.5 to 2.5 seconds regardless of process count, accounting for **48.2%** of total execution time at 12 processes (Amdahl's Law).

---

## 5. Kernel-Size Scaling (Mode `kernels`)

Evaluated with Hybrid $(P=3, T=4)$ on the $20,000 \times 20,000$ PGM across all 9 configurations (`results/final_summary_kernels.csv`, ratio of medians):

| Filter | Size ($k$) | Median Seq $T_{comp}$ (s) | Raw Range Seq (s) | Median Hybrid $T_{comp}$ (s) | Raw Range Hybrid (s) | Speedup (Ratio of Medians) |
|---|---|---|---|---|---|---|
| **Gaussian** | 3x3 | 4.830 s | [4.825, 6.100] | 1.436 s | [1.400, 2.636] | **3.364x** |
| **Gaussian** | 5x5 | 9.920 s | [9.895, 10.305] | 2.861 s | [2.323, 2.874] | **3.467x** |
| **Gaussian** | 7x7 | 17.417 s | [17.184, 17.660] | 4.206 s | [4.091, 4.367] | **4.141x** |
| **Sharpen** | 3x3 | 4.873 s | [4.722, 4.975] | 1.450 s | [1.360, 1.665] | **3.361x** |
| **Sharpen** | 5x5 | 10.003 s | [9.904, 10.050] | 2.623 s | [2.360, 2.631] | **3.814x** |
| **Sharpen** | 7x7 | 17.317 s | [17.145, 17.354] | 4.205 s | [3.818, 4.494] | **4.118x** |
| **Sobel** | 3x3 | 11.119 s | [11.027, 14.325] | 2.608 s | [2.508, 5.312] | **4.263x** |
| **Sobel** | 5x5 | 20.926 s | [20.917, 49.307] | 4.908 s | [4.351, 11.082] | **4.264x** |
| **Sobel** | 7x7 | 35.218 s | [35.135, 71.460] | 7.344 s | [7.168, 8.652] | **4.795x** |

### Assessment of Kernel-Size Effect
- **Observed Trend:** Speedup increases with kernel size across all three filters:
  - Gaussian: 3.36x (3x3) $\to$ 3.47x (5x5) $\to$ 4.14x (7x7)
  - Sharpen: 3.36x (3x3) $\to$ 3.81x (5x5) $\to$ 4.12x (7x7)
  - Sobel: 4.26x (3x3) $\to$ 4.26x (5x5) $\to$ 4.80x (7x7)
- **Significance:** The upward trend is consistent across all three filters. However, most individual step changes (e.g., 3.36x to 3.47x or 4.26x to 4.26x) fall well within the 20% measurement resolution limit, and run order in this benchmark was fixed. Therefore, while suggestive of higher arithmetic intensity improving parallel scaling, this trend is **suggestive, not conclusive**.

---

## 6. Limitations of the Study

1. **Occasional Run Outliers:** Sequential runs show occasional 2–2.5x outliers (e.g., 28.0 s in `final_main.csv`, 24.7 s in `final_order.csv`, and 49.3 s / 71.5 s during initial Sobel runs in `final_kernels.csv`). Parallel runs also occasionally show 2x outliers (e.g., 5.4 s vs 2.1 s for hybrid 3x4). Medians are used throughout this study to maintain robustness; the underlying cause of these outliers is unknown.
2. **Fixed Run Order in Main Benchmark:** In the main scaling benchmark, configurations were executed in fixed order (Sequential $\to$ OpenMP $\to$ MPI $\to$ Hybrid), meaning later runs may have run under different operating states. The randomised confirmation run addressed this limitation.
3. **Measurement Resolution Threshold:** Because the slowest measurement of the same configuration is 47% above the fastest, differences below approximately 20% cannot be resolved on this platform.
4. **Single-Node Execution:** All MPI ranks and OpenMP threads executed on one physical laptop sharing a single memory bus; true distributed-network interconnect latencies were not evaluated.
5. **Serial I/O Amdahl Cap:** Rank 0 executes single-process POSIX file I/O, which places an upper limit on overall wall-clock speedup regardless of compute parallelism.

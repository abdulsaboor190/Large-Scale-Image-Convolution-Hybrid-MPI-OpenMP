# Phase 5: Final Results and Performance Analysis

## 1. Experimental Environment and Methodology

- **Hardware Platform:** AMD Ryzen 5 7430U (6 physical cores, 12 hardware threads with SMT), ~4 GB RAM visible in WSL2 on Windows 11.
- **Benchmark Target:** $20,000 \times 20,000$ 8-bit grayscale PGM (`data/large/big_20000.pgm`, 400 MB).
- **Speedup Definition:**
  To maintain a single consistent, robust metric that is resilient to single-run outliers across long benchmark sessions, speedup is defined as the **ratio of medians**:
  $$\text{Speedup} = \frac{\text{median}(T_{seq, compute})}{\text{median}(T_{parallel, compute})}$$
  over the same run group.
- **Variability and Spread:**
  Raw per-configuration $[T_{min}, T_{max}]$ compute times (in seconds, not ratio ranges) are reported to capture the observed hardware execution spread.
- **Efficiency:**
  - Per hardware thread: $\text{Eff}_{hw} = \frac{\text{Speedup}}{P \times T}$
  - Per physical core: $\text{Eff}_{core} = \frac{\text{Speedup}}{\min(P \times T, 6)}$

---

## 2. Measurement Uncertainty and Resolution Limits

Laptop hardware platforms exhibit significant timing sensitivity to thermal state, battery/charger profile, background host tasks, and CPU power budgeting.

### Empirical Evidence for the 20% Resolution Threshold
In our benchmarks, the exact same configuration (**Gaussian 5x5, Hybrid $P=3, T=4$**) was measured across three separate sessions on this laptop:
1. **Phase 4 quick verification test:** **2.259 s** (cool initial system state)
2. **Phase 5 kernels benchmark:** **2.861 s**
3. **Phase 5 main benchmark:** **3.323 s** (sustained execution load)

Comparing these three runs shows that execution time for identical code on identical data varies by up to:
$$\frac{3.323 - 2.259}{3.323} \approx 32\%$$
Therefore, **performance differences below approximately 20% cannot be reliably resolved** from thermal, power, and OS scheduling jitter on this platform. Any performance difference below 20% is treated as an inconclusive tie.

### Sequential Baseline Spread
In the main benchmark (5 repetitions of Gaussian 5x5):
- Minimum: **10.121 s**
- Median: **11.203 s**
- Maximum: **28.029 s** (a single 28 s outlier; cause unproven, four other runs clustered at 10.1–11.5 s)

### Headline 12-Thread Speedup Range
Parallel scaling on this machine must be reported as a range reflecting operating conditions:
- **Under sustained execution load (main benchmark):** Speedup $\approx \mathbf{3.5\text{x}}$ (ratio of medians: 11.203 s / 3.141–3.323 s).
- **Under short, cool initial state (Phase 4 test):** Speedup up to $\mathbf{5.6\text{x}}$ ($T_{seq} = 12.678$ s vs $T_{hyb} = 2.259$ s).
- *Thermal and power state directly changes the measured result.*

---

## 3. Gaussian 5x5 Scaling Results (Main Benchmark)

Data from `results/final_summary_main.csv` (ratio of medians against sequential median $T_{comp} = 11.203$ s):

| Implementation | Procs ($P$) | Threads ($T$) | Total ($P \times T$) | Med $T_{comp}$ (s) | Raw Range $T_{comp}$ (s) | Speedup (Ratio of Medians) | Eff. / HW Thread | Eff. / Core | Med $T_{total}$ (s) |
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

---

## 4. Key Findings

### 4.1. Comparison of 12-Thread Configurations ($P \times T = 12$)
In the main benchmark, all 12-thread configurations achieved median compute times between **2.836 s and 3.620 s** (speedups between 3.10x and 3.95x):
- **OpenMP (1x12):** Median $T_{comp} = 2.836$ s (speedup 3.95x)
- **MPI (12x1):** Median $T_{comp} = 3.098$ s (speedup 3.62x)
- **Hybrid 4x3:** Median $T_{comp} = 3.141$ s (speedup 3.57x)
- **Hybrid 2x6:** Median $T_{comp} = 3.176$ s (speedup 3.53x)
- **Hybrid 3x4:** Median $T_{comp} = 3.323$ s (speedup 3.37x)
- **Hybrid 6x2:** Median $T_{comp} = 3.620$ s (speedup 3.10x)

**Methodological Caution Regarding 6x2:**
In the main benchmark, run order within each repetition was strictly fixed: sequential ran first, followed by OpenMP, MPI, and hybrid, with Hybrid 6x2 always executing last when the laptop was at its hottest. Therefore, the slower median for 6x2 in the main run (3.62 s vs 3.14–3.32 s) cannot be asserted as an intrinsic algorithmic deficiency; thermal accumulation across the fixed run sequence is a major confounding factor.

### 4.2. SMT Scaling (6 vs 12 Threads)
- Scaling from 1 to 6 physical cores reduces median compute time from 11.20 s to ~3.74 s (speedup ~3.0x).
- Scaling from 6 physical cores to 12 SMT threads provides an additional throughput gain, reaching median compute times of 2.84–3.14 s (speedup ~3.5x–3.9x, an additional ~20–25% throughput).
- *Hypothesis (not measured):* Because simultaneous multi-threading (SMT) shares execution units, L1/L2 caches, and memory access paths on each physical core, SMT threads provide incremental overlap rather than doubled throughput on memory-streaming convolution loops.

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

- **Halo Exchange Overhead:** Remains negligible ($< 0.002$ s, $< 0.1\%$ of compute time) across all runs because each rank exchanges only $2 \times 20,000$ bytes locally over RAM via shared-memory OpenMPI transports.
- **Serial Portion:** Single-process disk I/O, `MPI_Scatterv`, and `MPI_Gatherv` take approximately 1.5 to 2.5 seconds regardless of process count. At 12 processes, non-compute sequential overhead accounts for **48.2%** of total execution time. As dictated by Amdahl's Law, even if parallel compute time were driven to 0, total speedup cannot exceed $\sim 5$x without parallel I/O.

---

## 6. Kernel-Size Scaling (Mode `kernels`)

Evaluated with Hybrid $(P=3, T=4)$ on the $20,000 \times 20,000$ PGM across all 9 configurations (from `results/final_summary_kernels.csv`, ratio of medians):

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

- **Observation:** Speedup consistently increases with kernel size $k$ across all three filter types (from ~3.4x at 3x3 to ~4.1x–4.8x at 7x7).
- *Hypothesis (not measured):* Larger kernels perform $k^2$ multiply-accumulate operations per input byte read, increasing arithmetic intensity. With higher arithmetic intensity, computational pipelines may achieve better utilization relative to raw DRAM streaming limits.

---

## 7. Randomised Confirmation Run

To eliminate the ordering bias of the main benchmark (where run order was fixed and Hybrid 6x2 always ran last), an independent confirmation benchmark was executed with:
- 6 repetitions of Gaussian 5x5 on $20,000 \times 20,000$ image (`results/final_order.csv`).
- 9 configurations shuffled in **random order** per repetition with logged seeds.
- **15-second cooling pause** enforced between every run.

### Summary Table (`results/final_order_summary.csv`)
Sequential median $T_{comp} = 10.053$ s ([9.857, 24.667]):

| Configuration | Procs ($P$) | Threads ($T$) | Total ($P \times T$) | Med $T_{comp}$ (s) | Raw Range $T_{comp}$ (s) | Speedup (Ratio of Medians) | Med $T_{total}$ (s) |
|---|---|---|---|---|---|---|---|
| **Sequential** | 1 | 1 | 1 | 10.053 | [9.857, 24.667] | 1.000x | 11.053 |
| **OpenMP** | 1 | 6 | 6 | 3.050 | [2.914, 4.127] | 3.296x | 4.101 |
| **MPI** | 6 | 1 | 6 | 3.178 | [2.895, 3.594] | 3.164x | 5.327 |
| **MPI** | 12 | 1 | 12 | 2.111 | [2.015, 4.795] | 4.761x | 4.962 |
| **OpenMP** | 1 | 12 | 12 | 2.157 | [2.037, 2.545] | 4.659x | 3.493 |
| **Hybrid** | 2 | 6 | 12 | 2.197 | [2.138, 2.670] | 4.575x | 3.868 |
| **Hybrid** | 4 | 3 | 12 | 2.248 | [2.029, 2.633] | 4.472x | 4.205 |
| **Hybrid** | 3 | 4 | 12 | 2.252 | [2.070, 5.400] | 4.465x | 4.406 |
| **Hybrid** | 6 | 2 | 12 | 2.306 | [2.026, 2.653] | 4.358x | 4.386 |

### Comparison with Main Benchmark
1. **Agreement on 12-Thread Parity:**
   In the randomised test with cooling, all six 12-thread configurations clustered within an exceptionally tight median band of **2.111 s to 2.306 s** (speedup 4.36x to 4.76x). The total spread between the fastest (MPI 12x1 at 2.111 s) and the slowest (Hybrid 6x2 at 2.306 s) is only **0.195 s (8.4%)**, which is far below our 20% resolution threshold.
2. **Status of Hybrid 6x2:**
   When evaluated in randomised order with 15 s cooling, Hybrid 6x2 ran at **2.306 s median** (minimum 2.026 s), matching all other 12-thread configurations within 8.4%. This confirms that the apparent degradation of 6x2 in the main benchmark was largely an artifact of fixed run order (running last under accumulated heat), rather than a genuine algorithmic deficiency.
3. **Thermal Influence on Absolute Speedup:**
   With 15 s inter-run cooling, 12-thread parallel compute times improved from ~2.8–3.6 s to ~2.1–2.3 s, raising the ratio-of-medians speedup to ~4.4x–4.8x. This further corroborates that operating temperature and power states govern observed performance on this laptop.

---

## 8. Limitations of the Study

1. **Fixed Run Order in Main Benchmark:** In the main scaling benchmark, configurations were executed in a fixed order (Sequential $\to$ OpenMP $\to$ MPI $\to$ Hybrid), meaning later runs consistently encountered a warmer machine. The randomised confirmation run demonstrated that this order artifact accounted for the apparent deficit in Hybrid 6x2.
2. **Thermal & Power Budgets:** The laptop processor operates within a compact 15–25W TDP envelope. Sustained multi-core execution induces thermal and power state adjustments that fluctuate between runs.
3. **Resolution Limit:** Because identical code on identical data varies by up to ~32% across cool vs warm sessions, differences below ~20% cannot be resolved on this platform.
4. **Single-Node Shared Hardware:** All MPI ranks and OpenMP threads executed on one physical laptop sharing a single memory bus; distributed-network latencies were not measured.
5. **Serial I/O Amdahl Cap:** Rank 0 executes single-process POSIX I/O and root-driven scatter/gather, which limits end-to-end wall-clock speedup regardless of compute core count.

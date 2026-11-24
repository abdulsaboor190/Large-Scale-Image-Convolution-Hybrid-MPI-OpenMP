# Large-Scale Image Convolution: Sequential, OpenMP, MPI & Hybrid Parallelism

Parallel and Distributed Computing semester project at FAST-NUCES Karachi (BCS-7B).

This project implements and evaluates 2D image convolution (Sobel edge detection, Gaussian blur, Laplacian sharpening with 3x3, 5x5, 7x7 kernels) on large 8-bit grayscale images up to $20,000 \times 20,000$ (400 MB), parallelised with:
1. **OpenMP** (Shared-memory multi-threading)
2. **MPI** (1-D row-block domain decomposition with blocking halo exchange)
3. **Hybrid MPI + OpenMP** (`MPI_THREAD_FUNNELED`, multi-threaded ranks)

All parallel implementations produce **100% bit-exact outputs (`--tol 0`)** matching the sequential baseline.

---

## Quick Start & Build

### Requirements
- Ubuntu Linux (tested on WSL2 Ubuntu / Windows 11)
- GCC with OpenMP support (`gcc`, `mpicc`)
- OpenMPI (`mpirun`)
- Python 3 with `numpy`, `matplotlib` (in `~/pdc-venv` or system)

### Building
```bash
make clean && make check
```
Builds all binaries in `build/` and runs smoke tests.

### Running Programs
- **Sequential:**
  ```bash
  ./build/seq_convolve in.pgm out.pgm <sobel|gaussian|sharpen> <3|5|7>
  ```
- **OpenMP:**
  ```bash
  ./build/omp_convolve in.pgm out.pgm <sobel|gaussian|sharpen> <3|5|7> [threads]
  ```
- **Pure MPI:**
  ```bash
  mpirun -np P ./build/mpi_convolve in.pgm out.pgm <sobel|gaussian|sharpen> <3|5|7> 1
  ```
- **Hybrid MPI + OpenMP:**
  ```bash
  OMP_PLACES=threads OMP_PROC_BIND=close \
  mpirun --use-hwthread-cpus --bind-to hwthread --map-by socket:PE=T -np P \
  ./build/mpi_convolve in.pgm out.pgm <sobel|gaussian|sharpen> <3|5|7> T
  ```

---

## Test Suites & Verification

To verify all 1,244 test cases across all implementations:

```bash
source ~/pdc-venv/bin/activate

# 1. Sequential reference checks (54 cases)
bash scripts/run_tests.sh

# 2. OpenMP thread scaling correctness (--tol 0, 324 cases)
bash scripts/run_tests_omp.sh

# 3. Pure MPI correctness + stress tests (--tol 0, 444 cases)
bash scripts/run_tests_mpi.sh

# 4. Hybrid MPI+OpenMP correctness + stress tests (--tol 0, 420 cases)
bash scripts/run_tests_hybrid.sh
```

---

## Reproducing the Final Benchmarks

Ensure the $20,000 \times 20,000$ dataset exists:
```bash
python3 scripts/gen_dataset.py --size 20000 --out data/large/big_20000.pgm
```

Run the final benchmarks:
```bash
# 1. Main scaling benchmark (Gaussian 5x5, 5 repetitions, interleaved)
bash scripts/bench_final.sh main

# 2. Kernel-size scaling benchmark (all 9 filter/k combinations, 3 repetitions)
bash scripts/bench_final.sh kernels

# 3. Generate analysis CSVs and high-resolution plots
python3 scripts/analyze_final.py
python3 scripts/plot_final.py
```

---

## Documentation & Reports

- **Final Report:** [report/report.md](report/report.md)
- **Viva Preparation Guide (20 Questions & Answers):** [report/viva_notes.md](report/viva_notes.md)
- **Detailed Performance Analysis:** [docs/results_analysis.md](docs/results_analysis.md)
- **Architectural Specifications:**
  - Contract & Boundary Rules: [docs/DECISIONS.md](docs/DECISIONS.md)
  - OpenMP Schedule Tuning: [docs/omp_tuning.md](docs/omp_tuning.md)
  - OpenMP Vectorization & Restrict Fix: [docs/omp_fix.md](docs/omp_fix.md)
  - MPI Decomposition & Halo Exchange: [docs/mpi_design.md](docs/mpi_design.md)
  - Hybrid Threading Architecture: [docs/hybrid_design.md](docs/hybrid_design.md)

# Large-Scale Image Convolution: Hybrid MPI + OpenMP

NUCES Karachi, Parallel and Distributed Computing (BCS-7B). Phase 0 is complete: environment check, repo layout, kernels, test set and dataset generators. See `docs/DECISIONS.md` for the choices behind them.

## Layout

```
src/        kernels.h (kernel definitions), hello_hybrid.c (MPI+OpenMP smoke test), test_kernels.c
scripts/    check_env.sh, gen_test_images.py, gen_dataset.py, convert_to_pgm.py,
            compare_images.py, verify_kernels.py, refconv.py (NumPy reference), pgmio.py
data/test/  small inputs + data/test/ref/ expected outputs (committed)
data/large/ big generated images (git-ignored)
results/    timing CSVs and graphs (Phase 5)
report/     final report
docs/       DECISIONS.md
```

## Quick start

```bash
bash scripts/check_env.sh        # are gcc, mpicc, mpirun, python3, numpy, Pillow, matplotlib installed?
make check                       # builds, runs 2 ranks x 2 threads, verifies C kernels == Python kernels
python3 scripts/gen_test_images.py
```

Install on Ubuntu / WSL2:
```bash
sudo apt update && sudo apt install build-essential openmpi-bin libopenmpi-dev python3-pip
pip install numpy pillow matplotlib
```
Windows: use WSL2 with Ubuntu. macOS: `brew install gcc open-mpi`.

## Generating data

```bash
# Dataset A: one 20,000 x 20,000 grayscale image (400 MB, about 20 s)
python3 scripts/gen_dataset.py --size 20000 --out data/large/big_20000.pgm

# Dataset B: batch of 1,000 images of 2048 x 2048 (about 4 GB)
python3 scripts/gen_dataset.py --count 1000 --size 2048 --out data/large/batch/

# Optional: convert a real image
python3 scripts/convert_to_pgm.py some_satellite_image.tif data/large/real.pgm
```

## Correctness checking (used from Phase 1 onward)

```bash
python3 scripts/compare_images.py my_output.pgm data/test/ref/shapes_256__sobel_3x3.pgm --tol 1   # sequential vs reference
python3 scripts/compare_images.py parallel_out.pgm sequential_out.pgm --tol 0                      # parallel vs sequential: must be exact
```

Reference outputs are named `<image>__<filter>_<k>x<k>.pgm` with filters `sobel`, `gaussian`, `sharpen` and sizes 3, 5, 7.

## Notes

- If `mpirun` refuses to start because you are root (containers, some VMs), set `OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`. Normal user accounts do not need this.
- `mpirun --oversubscribe` lets you test with more ranks than cores. Use it for correctness only, never for timing.
- Set threads with `OMP_NUM_THREADS`, for example `OMP_NUM_THREADS=4 mpirun -np 2 ./build/<program>`.

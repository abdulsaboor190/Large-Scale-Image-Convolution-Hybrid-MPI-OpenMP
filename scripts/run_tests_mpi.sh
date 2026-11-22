#!/usr/bin/env bash
# Pure MPI correctness:
# 1. 324 standard cases: every data/test/*.pgm x 3 filters x 3 sizes x P in {1, 2, 3, 4, 7, 12}
#    compared with build/seq_convolve at --tol 0.
# 2. 120 stress test cases: generated random images with width 17 and heights in {1, 2, 3, 5, 8, 13}
#    x filters {sobel, gaussian} x sizes {3, 7} x P in {2, 3, 4, 7, 12} compared with seq_convolve at --tol 0.
#
# Temp files are stored in a temporary directory and cleaned up on exit.
# Exit code is non-zero if any test fails.

cd "$(dirname "$0")/.."

if [[ -f "$HOME/pdc-venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/pdc-venv/bin/activate"
fi

SEQ=./build/seq_convolve
MPI=./build/mpi_convolve

if [[ ! -x "$SEQ" || ! -x "$MPI" ]]; then
    echo "missing $SEQ or $MPI; run make first" >&2
    exit 1
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

echo "=== PART 1: Standard MPI Correctness (324 cases) ==="
pass=0
fail=0

for img in data/test/*.pgm; do
    base=$(basename "$img" .pgm)
    for filter in sobel gaussian sharpen; do
        for k in 3 5 7; do
            seq_out="$tmpdir/${base}__${filter}_${k}x${k}__seq.pgm"
            name="$base $filter ${k}x${k}"
            if ! "$SEQ" "$img" "$seq_out" "$filter" "$k" >/dev/null 2>&1; then
                echo "FAIL $name (seq_convolve)"
                fail=$((fail + 6))
                continue
            fi

            for procs in 1 2 3 4 7 12; do
                mpi_out="$tmpdir/${base}__${filter}_${k}x${k}__p${procs}.pgm"
                case_name="$name procs=$procs"
                if ! mpirun --oversubscribe -np "$procs" "$MPI" "$img" "$mpi_out" "$filter" "$k" >/dev/null 2>&1; then
                    echo "FAIL $case_name (mpi_convolve execution failed)"
                    fail=$((fail + 1))
                    rm -f "$mpi_out"
                    continue
                fi

                if python3 scripts/compare_images.py "$mpi_out" "$seq_out" --tol 0 >/dev/null 2>&1; then
                    echo "PASS $case_name"
                    pass=$((pass + 1))
                else
                    echo "FAIL $case_name (mismatch against sequential)"
                    fail=$((fail + 1))
                fi
                rm -f "$mpi_out"
            done
            rm -f "$seq_out"
        done
    done
done

echo ""
echo "Part 1 Summary: $pass passed, $fail failed (expected 324)"
if [[ $fail -ne 0 ]]; then
    echo "FAILED in Part 1" >&2
    exit 1
fi

echo ""
echo "=== PART 2: Stress Testing (120 cases: awkward heights & small blocks) ==="
stress_pass=0
stress_fail=0

# Python script to generate random 8-bit grayscale PGM
python3 -c "
import numpy as np
import sys

def write_pgm(path, w, h, seed):
    rng = np.random.RandomState(seed)
    data = rng.randint(0, 256, size=(h, w), dtype=np.uint8)
    with open(path, 'wb') as f:
        header = f'P5\n{w} {h}\n255\n'.encode('ascii')
        f.write(header)
        f.write(data.tobytes())

seed = 42
for h in [1, 2, 3, 5, 8, 13]:
    p = f'$tmpdir/rand_17x{h}.pgm'
    write_pgm(p, 17, h, seed)
    seed += 1
"

for h in 1 2 3 5 8 13; do
    stress_img="$tmpdir/rand_17x${h}.pgm"
    base="rand_17x${h}"
    for filter in sobel gaussian; do
        for k in 3 7; do
            seq_out="$tmpdir/${base}__${filter}_${k}x${k}__seq.pgm"
            name="$base $filter ${k}x${k}"
            if ! "$SEQ" "$stress_img" "$seq_out" "$filter" "$k" >/dev/null 2>&1; then
                echo "FAIL $name (seq_convolve)"
                stress_fail=$((stress_fail + 5))
                continue
            fi

            for procs in 2 3 4 7 12; do
                mpi_out="$tmpdir/${base}__${filter}_${k}x${k}__p${procs}.pgm"
                case_name="$name procs=$procs"
                if ! mpirun --oversubscribe -np "$procs" "$MPI" "$stress_img" "$mpi_out" "$filter" "$k" >/dev/null 2>&1; then
                    echo "FAIL $case_name (mpi_convolve execution failed)"
                    stress_fail=$((stress_fail + 1))
                    rm -f "$mpi_out"
                    continue
                fi

                if python3 scripts/compare_images.py "$mpi_out" "$seq_out" --tol 0 >/dev/null 2>&1; then
                    echo "PASS $case_name"
                    stress_pass=$((stress_pass + 1))
                else
                    echo "FAIL $case_name (mismatch against sequential)"
                    stress_fail=$((stress_fail + 1))
                fi
                rm -f "$mpi_out"
            done
            rm -f "$seq_out"
        done
    done
done

echo ""
echo "Part 2 Summary: $stress_pass passed, $stress_fail failed (expected 120)"
echo "Total MPI Tests: $((pass + stress_pass)) passed, $((fail + stress_fail)) failed (expected 444)"

if [[ $fail -ne 0 || $stress_fail -ne 0 ]]; then
    exit 1
fi
exit 0

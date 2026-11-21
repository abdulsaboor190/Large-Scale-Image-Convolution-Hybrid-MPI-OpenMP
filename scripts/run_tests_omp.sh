#!/usr/bin/env bash
# OpenMP correctness: every data/test image x 3 filters x 3 sizes x thread
# counts 1, 2, 3, 4, 7, 12 must match build/seq_convolve at --tol 0.
# Odd counts, and counts larger than a tiny image's height, are deliberate.
# Temp files live in a directory that is removed on exit.
# Exit status is non-zero if any case fails.
cd "$(dirname "$0")/.."

if [[ -f "$HOME/pdc-venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/pdc-venv/bin/activate"
fi

SEQ=./build/seq_convolve
OMP=./build/omp_convolve
if [[ ! -x "$SEQ" || ! -x "$OMP" ]]; then
    echo "missing $SEQ or $OMP; run make first" >&2
    exit 1
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

pass=0
fail=0
for img in data/test/*.pgm; do
    base=$(basename "$img" .pgm)
    for filter in sobel gaussian sharpen; do
        for k in 3 5 7; do
            seq_out="$tmpdir/${base}__${filter}_${k}x${k}__seq.pgm"
            name="$base $filter ${k}x${k}"
            if ! "$SEQ" "$img" "$seq_out" "$filter" "$k" >/dev/null; then
                echo "FAIL $name (seq_convolve)"
                fail=$((fail + 6))
                continue
            fi
            for threads in 1 2 3 4 7 12; do
                omp_out="$tmpdir/${base}__${filter}_${k}x${k}__t${threads}.pgm"
                case_name="$name threads=$threads"
                if ! "$OMP" "$img" "$omp_out" "$filter" "$k" "$threads" >/dev/null; then
                    echo "FAIL $case_name (omp_convolve)"
                    fail=$((fail + 1))
                    rm -f "$omp_out"
                    continue
                fi
                if python3 scripts/compare_images.py "$omp_out" "$seq_out" --tol 0 >/dev/null; then
                    echo "PASS $case_name"
                    pass=$((pass + 1))
                else
                    echo "FAIL $case_name"
                    python3 scripts/compare_images.py "$omp_out" "$seq_out" --tol 0
                    fail=$((fail + 1))
                fi
                rm -f "$omp_out"
            done
            rm -f "$seq_out"
        done
    done
done

echo "summary: $pass passed, $fail failed, $((pass + fail)) total"
if [[ "$fail" -ne 0 ]]; then
    exit 1
fi
exit 0

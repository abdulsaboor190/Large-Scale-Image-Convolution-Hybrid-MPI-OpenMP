#!/usr/bin/env bash
# Correctness suite for the sequential baseline.
# Every data/test image x {sobel,gaussian,sharpen} x {3,5,7} is compared
# with data/test/ref/<image>__<filter>_<k>x<k>.pgm at tolerance 1
# (float32 C vs float64 NumPy; see docs/DECISIONS.md).
# Temp outputs are removed on exit. Exit status is non-zero if any case fails.
cd "$(dirname "$0")/.."

BIN=./build/seq_convolve
if [[ ! -x "$BIN" ]]; then
    echo "missing $BIN; run make first" >&2
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
            out="$tmpdir/${base}__${filter}_${k}x${k}.pgm"
            ref="data/test/ref/${base}__${filter}_${k}x${k}.pgm"
            name="$base $filter ${k}x${k}"
            if ! "$BIN" "$img" "$out" "$filter" "$k" >/dev/null; then
                echo "FAIL $name (seq_convolve)"
                fail=$((fail + 1))
                continue
            fi
            if python3 scripts/compare_images.py "$out" "$ref" --tol 1; then
                echo "PASS $name"
                pass=$((pass + 1))
            else
                echo "FAIL $name"
                fail=$((fail + 1))
            fi
        done
    done
done

echo "summary: $pass passed, $fail failed, $((pass + fail)) total"
if [[ "$fail" -ne 0 ]]; then
    exit 1
fi
exit 0

#!/usr/bin/env bash
# Sequential baseline on data/large/big_20000.pgm.
# 3 filters x 3 kernel sizes x 3 repeats. Appends to results/baseline_seq.csv
# (header written once). The output image is a temp file, deleted after each run.
# Pins the process to core 4 when taskset is available, so the timing is one core.
cd "$(dirname "$0")/.."
set -euo pipefail

# The binary is linked with OpenMP, but this baseline must stay one thread.
# There are no parallel regions in seq_convolve.c; this makes that explicit.
export OMP_NUM_THREADS=1

IMG=data/large/big_20000.pgm
BIN=./build/seq_convolve
CSV=results/baseline_seq.csv

if [[ ! -f "$IMG" ]]; then
    echo "missing $IMG" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "missing $BIN; run make first" >&2
    exit 1
fi

mkdir -p results
echo "filter,k,run,width,height,compute_s,total_s" > "$CSV"

if command -v taskset >/dev/null 2>&1; then
    echo "pinning to core 4" >&2
    run_bin() { taskset -c 4 "$BIN" "$@"; }
else
    echo "taskset not found; running unpinned" >&2
    run_bin() { "$BIN" "$@"; }
fi

for filter in sobel gaussian sharpen; do
    for k in 3 5 7; do
        for run in 1 2 3; do
            tmp=$(mktemp)
            echo "run filter=$filter k=$k repeat=$run" >&2
            if ! line=$(run_bin "$IMG" "$tmp" "$filter" "$k"); then
                echo "seq_convolve failed for $filter $k run $run" >&2
                rm -f "$tmp"
                exit 1
            fi
            rm -f "$tmp"
            w=$(printf '%s\n' "$line" | sed -n 's/.*w=\([0-9]*\).*/\1/p')
            h=$(printf '%s\n' "$line" | sed -n 's/.* h=\([0-9]*\).*/\1/p')
            compute=$(printf '%s\n' "$line" | sed -n 's/.*compute_s=\([0-9.][0-9.]*\).*/\1/p')
            total=$(printf '%s\n' "$line" | sed -n 's/.*total_s=\([0-9.][0-9.]*\).*/\1/p')
            if [[ -z "$compute" || -z "$total" || -z "$w" || -z "$h" ]]; then
                echo "could not parse stdout: $line" >&2
                exit 1
            fi
            echo "$filter,$k,$run,$w,$h,$compute,$total" >> "$CSV"
            echo "$line" >&2
        done
    done
done

echo "appended 27 rows to $CSV" >&2

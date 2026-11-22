#!/usr/bin/env bash
# Thread scaling on data/large/big_20000.pgm.
# threads 1, 2, 3, 4, 6, 8, 12 x sobel/gaussian/sharpen x 3/5/7, 3 repeats.
# Uses the schedule compiled into omp_convolve (OMP_SCHEDULE is unset).
# Binding: OMP_PLACES=cores OMP_PROC_BIND=close.
# Appends to results/omp_scaling.csv (header written once).
cd "$(dirname "$0")/.."
set -euo pipefail

export OMP_PLACES=cores
export OMP_PROC_BIND=close
unset OMP_SCHEDULE || true

IMG=data/large/big_20000.pgm
BIN=./build/omp_convolve
CSV=results/omp_scaling.csv

if [[ ! -f "$IMG" ]]; then
    echo "missing $IMG" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "missing $BIN; run make first" >&2
    exit 1
fi

mkdir -p results
echo "filter,k,threads,run,width,height,compute_s,total_s" > "$CSV"

for threads in 1 2 3 4 6 8 12; do
    for filter in sobel gaussian sharpen; do
        for k in 3 5 7; do
            for run in 1 2 3; do
                tmp=$(mktemp)
                echo "bench threads=$threads filter=$filter k=$k run=$run" >&2
                if ! line=$("$BIN" "$IMG" "$tmp" "$filter" "$k" "$threads"); then
                    echo "omp_convolve failed" >&2
                    rm -f "$tmp"
                    exit 1
                fi
                rm -f "$tmp"
                w=$(printf '%s\n' "$line" | sed -n 's/.*w=\([0-9]*\).*/\1/p')
                h=$(printf '%s\n' "$line" | sed -n 's/.* h=\([0-9]*\).*/\1/p')
                got_threads=$(printf '%s\n' "$line" | sed -n 's/.*threads=\([0-9]*\).*/\1/p')
                compute=$(printf '%s\n' "$line" | sed -n 's/.*compute_s=\([0-9.][0-9.]*\).*/\1/p')
                total=$(printf '%s\n' "$line" | sed -n 's/.*total_s=\([0-9.][0-9.]*\).*/\1/p')
                if [[ -z "$compute" || -z "$total" || -z "$w" || -z "$h" || -z "$got_threads" ]]; then
                    echo "could not parse stdout: $line" >&2
                    exit 1
                fi
                if [[ "$got_threads" != "$threads" ]]; then
                    echo "asked for $threads threads but program reported $got_threads" >&2
                    exit 1
                fi
                echo "$filter,$k,$threads,$run,$w,$h,$compute,$total" >> "$CSV"
                echo "$line" >&2
            done
        done
    done
done

echo "appended scaling rows to $CSV" >&2

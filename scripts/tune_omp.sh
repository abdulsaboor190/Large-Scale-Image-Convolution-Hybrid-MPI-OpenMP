#!/usr/bin/env bash
# Schedule and chunk-size sweep on data/large/big_20000.pgm.
# gaussian 5x5 and sobel 7x7, 6 and 12 threads, 3 repeats.
# Schedules: static (default chunk), static 1/16/64,
# dynamic 1/16/64/256, guided.
# The program honours OMP_SCHEDULE because its loop is schedule(runtime).
# Appends to results/omp_schedule_tuning.csv (header written once).
cd "$(dirname "$0")/.."
set -euo pipefail

export OMP_PLACES=cores
export OMP_PROC_BIND=close

IMG=data/large/big_20000.pgm
BIN=./build/omp_convolve
CSV=results/omp_schedule_tuning.csv

if [[ ! -f "$IMG" ]]; then
    echo "missing $IMG" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "missing $BIN; run make first" >&2
    exit 1
fi

mkdir -p results
if [[ ! -f "$CSV" ]]; then
    echo "filter,k,threads,schedule,chunk,run,compute_s,total_s" > "$CSV"
fi

# kind:chunk. An empty chunk means OMP_SCHEDULE=<kind> and csv chunk 0.
schedules=(
    "static:"
    "static:1"
    "static:16"
    "static:64"
    "dynamic:1"
    "dynamic:16"
    "dynamic:64"
    "dynamic:256"
    "guided:"
)

for filter_k in "gaussian 5" "sobel 7"; do
    filter=${filter_k% *}
    k=${filter_k#* }
    for threads in 6 12; do
        for spec in "${schedules[@]}"; do
            kind=${spec%%:*}
            chunk=${spec#*:}
            if [[ -z "$chunk" ]]; then
                export OMP_SCHEDULE="$kind"
                chunk_csv=0
            else
                export OMP_SCHEDULE="${kind},${chunk}"
                chunk_csv=$chunk
            fi
            for run in 1 2 3; do
                tmp=$(mktemp)
                echo "tune filter=$filter k=$k threads=$threads schedule=$OMP_SCHEDULE run=$run" >&2
                if ! line=$("$BIN" "$IMG" "$tmp" "$filter" "$k" "$threads"); then
                    echo "omp_convolve failed" >&2
                    rm -f "$tmp"
                    exit 1
                fi
                rm -f "$tmp"
                got=$(printf '%s\n' "$line" | sed -n 's/.*schedule=\([a-z]*\).*/\1/p')
                compute=$(printf '%s\n' "$line" | sed -n 's/.*compute_s=\([0-9.][0-9.]*\).*/\1/p')
                total=$(printf '%s\n' "$line" | sed -n 's/.*total_s=\([0-9.][0-9.]*\).*/\1/p')
                if [[ "$got" != "$kind" || -z "$compute" || -z "$total" ]]; then
                    echo "bad stdout (wanted schedule=$kind): $line" >&2
                    exit 1
                fi
                echo "$filter,$k,$threads,$kind,$chunk_csv,$run,$compute,$total" >> "$CSV"
                echo "$line" >&2
            done
        done
    done
done

echo "tuning rows appended to $CSV" >&2

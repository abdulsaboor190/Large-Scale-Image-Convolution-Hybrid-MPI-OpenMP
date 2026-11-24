#!/usr/bin/env bash
# scripts/bench_final.sh - Phase 5 Final Benchmark Runner
# Usage:
#   bash scripts/bench_final.sh main     # 5 repeats, gaussian 5x5, 21 runs/rep -> 105 rows + 1 header = 106 lines
#   bash scripts/bench_final.sh kernels  # 3 repeats, all 9 filter/k combos, seq + hyb(3,4) -> 54 rows + 1 header = 55 lines
#   bash scripts/bench_final.sh order    # 6 repeats, gaussian 5x5, 9 configs shuffled, 15s cool -> 54 rows + 1 header = 55 lines

cd "$(dirname "$0")/.."
set -euo pipefail

if [[ -f "$HOME/pdc-venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/pdc-venv/bin/activate"
fi

MODE="${1:-}"
if [[ "$MODE" != "main" && "$MODE" != "kernels" && "$MODE" != "order" ]]; then
    echo "Usage: $0 <main|kernels|order>" >&2
    exit 1
fi

IMG=data/large/big_20000.pgm
OUT=data/large/bench_out.pgm
LOG=results/final_bench_log.txt

if [[ ! -f "$IMG" ]]; then
    echo "missing $IMG; run gen_dataset.py first" >&2
    exit 1
fi

mkdir -p results data/large

# Helper to log system state at start of rep
log_system_state() {
    local rep="$1"
    local m="$2"
    {
        echo "=== Repetition $rep ($m) $(date -Iseconds) ==="
        uptime
        grep "cpu MHz" /proc/cpuinfo || true
        echo ""
    } >> "$LOG"
}

# Run sequential
# Args: group filter k rep [cooldown]
run_seq() {
    local group="$1" filt="$2" k="$3" rep="$4" cooldown="${5:-2}"
    rm -f "$OUT"
    local line
    line=$(taskset -c 4 ./build/seq_convolve "$IMG" "$OUT" "$filt" "$k")
    rm -f "$OUT"

    local cs ts ts_now
    cs=$(printf '%s\n' "$line" | sed -n 's/.*compute_s=\([0-9.]*\).*/\1/p')
    ts=$(printf '%s\n' "$line" | sed -n 's/.*total_s=\([0-9.]*\).*/\1/p')
    ts_now=$(date +%s)
    echo "$group,seq,$filt,$k,1,1,$rep,,,$cs,$cs,,$ts,$ts_now" >> "$CSV"
    sleep "$cooldown"
}

# Run OpenMP
# Args: group filter k threads rep [cooldown]
run_omp() {
    local group="$1" filt="$2" k="$3" thr="$4" rep="$5" cooldown="${6:-2}"
    rm -f "$OUT"
    local line
    line=$(OMP_PLACES=cores OMP_PROC_BIND=close ./build/omp_convolve "$IMG" "$OUT" "$filt" "$k" "$thr")
    rm -f "$OUT"

    local cs ts ts_now
    cs=$(printf '%s\n' "$line" | sed -n 's/.*compute_s=\([0-9.]*\).*/\1/p')
    ts=$(printf '%s\n' "$line" | sed -n 's/.*total_s=\([0-9.]*\).*/\1/p')
    ts_now=$(date +%s)
    echo "$group,omp,$filt,$k,1,$thr,$rep,,,$cs,$cs,,$ts,$ts_now" >> "$CSV"
    sleep "$cooldown"
}

# Run MPI / Hybrid
# Args: group mode filter k procs threads rep [cooldown]
run_mpi_hyb() {
    local group="$1" m="$2" filt="$3" k="$4" procs="$5" thr="$6" rep="$7" cooldown="${8:-2}"
    local total=$((procs * thr))
    rm -f "$OUT"
    local line

    if [[ "$total" -le 6 ]]; then
        # Physical-core recipe
        line=$(OMP_PLACES=cores OMP_PROC_BIND=close \
               mpirun --bind-to core --map-by socket:PE="$thr" -np "$procs" \
               ./build/mpi_convolve "$IMG" "$OUT" "$filt" "$k" "$thr")
    else
        # Hardware-thread SMT recipe
        line=$(OMP_PLACES=threads OMP_PROC_BIND=close \
               mpirun --use-hwthread-cpus --bind-to hwthread --map-by socket:PE="$thr" -np "$procs" \
               ./build/mpi_convolve "$IMG" "$OUT" "$filt" "$k" "$thr")
    fi
    rm -f "$OUT"

    local sc hl cv cp gt tt ts_now
    sc=$(printf '%s\n' "$line" | sed -n 's/.*scatter_s=\([0-9.]*\).*/\1/p')
    hl=$(printf '%s\n' "$line" | sed -n 's/.*halo_s=\([0-9.]*\).*/\1/p')
    cv=$(printf '%s\n' "$line" | sed -n 's/.*conv_s=\([0-9.]*\).*/\1/p')
    cp=$(printf '%s\n' "$line" | sed -n 's/.*compute_s=\([0-9.]*\).*/\1/p')
    gt=$(printf '%s\n' "$line" | sed -n 's/.*gather_s=\([0-9.]*\).*/\1/p')
    tt=$(printf '%s\n' "$line" | sed -n 's/.*total_s=\([0-9.]*\).*/\1/p')
    ts_now=$(date +%s)

    echo "$group,$m,$filt,$k,$procs,$thr,$rep,$sc,$hl,$cv,$cp,$gt,$tt,$ts_now" >> "$CSV"
    sleep "$cooldown"
}

if [[ "$MODE" == "main" ]]; then
    CSV=results/final_main.csv
    echo "group,mode,filter,k,procs,threads,rep,scatter_s,halo_s,conv_s,compute_s,gather_s,total_s,timestamp" > "$CSV"
    echo "Starting final benchmark: MODE main (5 repetitions, target: 106 lines in $CSV)" >&2

    for rep in 1 2 3 4 5; do
        echo "[Repetition $rep/5] Logging system state and beginning 21 runs..." >&2
        log_system_state "$rep" "main"

        # 1 sequential run
        run_seq "main" "gaussian" 5 "$rep"

        # OpenMP only (threads 1, 2, 3, 4, 6, 8, 12)
        for t in 1 2 3 4 6 8 12; do
            run_omp "main" "gaussian" 5 "$t" "$rep"
        done

        # MPI only (T=1, P=1, 2, 3, 4, 6, 8, 12)
        for p in 1 2 3 4 6 8 12; do
            run_mpi_hyb "main" "mpi" "gaussian" 5 "$p" 1 "$rep"
        done

        # Hybrid combos: (2,3), (3,2), (2,6), (3,4), (4,3), (6,2)
        for pt in "2 3" "3 2" "2 6" "3 4" "4 3" "6 2"; do
            read -r p t <<< "$pt"
            run_mpi_hyb "main" "hybrid" "gaussian" 5 "$p" "$t" "$rep"
        done

        echo "[Repetition $rep/5] Finished." >&2
    done
    echo "MODE main complete. Results in $CSV" >&2

elif [[ "$MODE" == "kernels" ]]; then
    CSV=results/final_kernels.csv
    echo "group,mode,filter,k,procs,threads,rep,scatter_s,halo_s,conv_s,compute_s,gather_s,total_s,timestamp" > "$CSV"
    echo "Starting final benchmark: MODE kernels (3 repetitions, target: 55 lines in $CSV)" >&2

    for rep in 1 2 3; do
        echo "[Repetition $rep/3] Logging system state and beginning 18 runs..." >&2
        log_system_state "$rep" "kernels"

        for filt in sobel gaussian sharpen; do
            for k in 3 5 7; do
                run_seq "kernels" "$filt" "$k" "$rep"
                run_mpi_hyb "kernels" "hybrid" "$filt" "$k" 3 4 "$rep"
            done
        done
        echo "[Repetition $rep/3] Finished." >&2
    done
    echo "MODE kernels complete. Results in $CSV" >&2

elif [[ "$MODE" == "order" ]]; then
    CSV=results/final_order.csv
    echo "group,mode,filter,k,procs,threads,rep,scatter_s,halo_s,conv_s,compute_s,gather_s,total_s,timestamp" > "$CSV"
    echo "Starting randomised confirmation benchmark: MODE order (6 repetitions, 9 configs shuffled, 15s cooldown, target: 55 lines in $CSV)" >&2

    for rep in 1 2 3 4 5 6; do
        seed=$(( 1000 * rep + ($(date +%s) % 1000) ))
        echo "[Repetition $rep/6] Logging system state (seed=$seed)..." >&2
        {
            echo "=== Repetition $rep (order) $(date -Iseconds) seed=$seed ==="
            uptime
            grep "cpu MHz" /proc/cpuinfo || true
            echo ""
        } >> "$LOG"

        # 9 configs to shuffle:
        # 1: seq
        # 2: omp 6
        # 3: omp 12
        # 4: mpi 6 1
        # 5: mpi 12 1
        # 6: hybrid 2 6
        # 7: hybrid 3 4
        # 8: hybrid 4 3
        # 9: hybrid 6 2
        order=$(python3 -c "import random; items = ['seq:1:1', 'omp:1:6', 'omp:1:12', 'mpi:6:1', 'mpi:12:1', 'hybrid:2:6', 'hybrid:3:4', 'hybrid:4:3', 'hybrid:6:2']; random.seed($seed); random.shuffle(items); print(' '.join(items))")
        echo "[Repetition $rep/6] Execution order: $order" >&2

        for item in $order; do
            IFS=':' read -r m p t <<< "$item"
            echo "  -> running $m (procs=$p, threads=$t) with 15s cooldown..." >&2
            if [[ "$m" == "seq" ]]; then
                run_seq "order" "gaussian" 5 "$rep" 15
            elif [[ "$m" == "omp" ]]; then
                run_omp "order" "gaussian" 5 "$t" "$rep" 15
            else
                run_mpi_hyb "order" "$m" "gaussian" 5 "$p" "$t" "$rep" 15
            fi
        done
        echo "[Repetition $rep/6] Finished." >&2
    done
    echo "MODE order complete. Results in $CSV" >&2
fi

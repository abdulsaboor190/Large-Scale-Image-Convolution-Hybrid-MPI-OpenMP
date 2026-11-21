#!/usr/bin/env python3
"""Median OpenMP compute time, speedup and efficiency.

Speedup = T_seq_median (results/baseline_seq_summary.csv) / T_omp_median.
Efficiency = speedup / threads.
Writes results/omp_summary.csv.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "results" / "baseline_seq_summary.csv"
OMP = ROOT / "results" / "omp_scaling.csv"
OUT = ROOT / "results" / "omp_summary.csv"

FILTERS = ("sobel", "gaussian", "sharpen")


def main():
    if not SEQ.is_file() or not OMP.is_file():
        print(f"missing {SEQ.name} or {OMP.name}", file=sys.stderr)
        return 1

    seq = {}
    with SEQ.open(newline="") as f:
        for row in csv.DictReader(f):
            seq[(row["filter"], int(row["k"]))] = float(row["median_compute_s"])

    groups = defaultdict(list)
    with OMP.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["filter"], int(row["k"]), int(row["threads"]))
            groups[key].append(float(row["compute_s"]))

    if not groups:
        print(f"{OMP} has no data rows", file=sys.stderr)
        return 1

    lines = []
    for filt in FILTERS:
        ks = sorted({k for (f, k, _t) in groups if f == filt})
        for k in ks:
            threads_list = sorted({t for (f, kk, t) in groups if f == filt and kk == k})
            for threads in threads_list:
                samples = groups[(filt, k, threads)]
                med = median(samples)
                if (filt, k) not in seq:
                    print(f"no sequential median for {filt} {k}", file=sys.stderr)
                    return 1
                speedup = seq[(filt, k)] / med
                efficiency = speedup / threads
                lines.append((filt, k, threads, med, speedup, efficiency, len(samples)))

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filter", "k", "threads", "median_compute_s", "speedup", "efficiency", "runs"])
        for filt, k, threads, med, speedup, efficiency, runs in lines:
            w.writerow([
                filt, k, threads,
                f"{med:.3f}", f"{speedup:.3f}", f"{efficiency:.3f}", runs,
            ])
    print(f"wrote {OUT} ({len(lines)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Speedup and efficiency vs thread count, from results/omp_summary.csv.

Writes results/omp_speedup.png and results/omp_efficiency.png.
One line per filter and kernel size, plus the ideal line.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "omp_summary.csv"
SPEEDUP_PNG = ROOT / "results" / "omp_speedup.png"
EFF_PNG = ROOT / "results" / "omp_efficiency.png"


def load():
    series = defaultdict(lambda: {"threads": [], "speedup": [], "efficiency": []})
    with SRC.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        key = (row["filter"], int(row["k"]))
        series[key]["threads"].append(int(row["threads"]))
        series[key]["speedup"].append(float(row["speedup"]))
        series[key]["efficiency"].append(float(row["efficiency"]))
    return series


def draw(series, field, ylabel, ideal, out_path):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    thread_max = 1
    for (filt, k) in sorted(series, key=lambda item: (item[0], item[1])):
        xs = series[(filt, k)]["threads"]
        ys = series[(filt, k)][field]
        thread_max = max(thread_max, max(xs) if xs else 1)
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{filt} {k}x{k}")
    if field == "speedup":
        ideal_x = list(range(1, thread_max + 1))
        ax.plot(ideal_x, ideal_x, linestyle="--", color="black", linewidth=1.0, label="ideal")
    else:
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0, label="ideal")
    ax.set_xlabel("threads")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({t for s in series.values() for t in s["threads"]}))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    if not SRC.is_file():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    series = load()
    if not series:
        print(f"{SRC} has no rows", file=sys.stderr)
        return 1
    draw(series, "speedup", "speedup (T_seq / T_omp)", None, SPEEDUP_PNG)
    draw(series, "efficiency", "efficiency (speedup / threads)", 1.0, EFF_PNG)
    return 0


if __name__ == "__main__":
    sys.exit(main())

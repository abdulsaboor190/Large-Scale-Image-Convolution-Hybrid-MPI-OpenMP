#!/usr/bin/env python3
# scripts/analyze_order.py - Analyzes randomised confirmation benchmark (results/final_order.csv)

import csv
from collections import defaultdict
import statistics

def analyze_order():
    rows = []
    with open('results/final_order.csv', 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    seq_compute = [float(r['compute_s']) for r in rows if r['mode'] == 'seq']
    seq_total = [float(r['total_s']) for r in rows if r['mode'] == 'seq']

    med_seq_comp = statistics.median(seq_compute)
    med_seq_tot = statistics.median(seq_total)

    groups = defaultdict(list)
    for r in rows:
        key = (r['mode'], int(r['procs']), int(r['threads']))
        groups[key].append({
            'compute_s': float(r['compute_s']),
            'total_s': float(r['total_s'])
        })

    summary_rows = []
    for (mode, p, t), entries in groups.items():
        comp_list = sorted([e['compute_s'] for e in entries])
        tot_list = sorted([e['total_s'] for e in entries])

        med_c = statistics.median(comp_list)
        min_c = comp_list[0]
        max_c = comp_list[-1]

        med_t = statistics.median(tot_list)
        min_t = tot_list[0]
        max_t = tot_list[-1]

        sp = med_seq_comp / med_c
        tot_sp = med_seq_tot / med_t

        tot_threads = p * t
        eff_hw = sp / tot_threads
        eff_core = sp / min(tot_threads, 6)

        summary_rows.append({
            'mode': mode,
            'procs': p,
            'threads': t,
            'total_threads': tot_threads,
            'med_compute_s': f"{med_c:.3f}",
            'min_compute_s': f"{min_c:.3f}",
            'max_compute_s': f"{max_c:.3f}",
            'med_total_s': f"{med_t:.3f}",
            'min_total_s': f"{min_t:.3f}",
            'max_total_s': f"{max_t:.3f}",
            'speedup': f"{sp:.3f}",
            'eff_hw_thread': f"{eff_hw:.3f}",
            'eff_core': f"{eff_core:.3f}",
            'total_speedup': f"{tot_sp:.3f}"
        })

    mode_order = {'seq': 0, 'omp': 1, 'mpi': 2, 'hybrid': 3}
    summary_rows.sort(key=lambda r: (mode_order[r['mode']], r['total_threads'], r['procs']))

    with open('results/final_order_summary.csv', 'w', newline='') as f:
        fieldnames = ['mode', 'procs', 'threads', 'total_threads',
                      'med_compute_s', 'min_compute_s', 'max_compute_s',
                      'med_total_s', 'min_total_s', 'max_total_s',
                      'speedup', 'eff_hw_thread', 'eff_core', 'total_speedup']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Wrote results/final_order_summary.csv")

if __name__ == '__main__':
    analyze_order()

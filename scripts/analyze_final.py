#!/usr/bin/env bash
# scripts/analyze_final.py - Generates final benchmark summaries

import csv
from collections import defaultdict

def analyze_main():
    rows = []
    with open('results/final_main.csv', 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 1. Map sequential compute_s per repetition
    seq_compute = {}
    seq_total = {}
    for r in rows:
        if r['mode'] == 'seq':
            rep = int(r['rep'])
            seq_compute[rep] = float(r['compute_s'])
            seq_total[rep] = float(r['total_s'])

    # Print sequential spread
    seq_vals = sorted(seq_compute.values())
    print(f"Sequential compute_s spread across 5 reps: min={seq_vals[0]:.3f}s, median={seq_vals[2]:.3f}s, max={seq_vals[-1]:.3f}s (ratio max/min = {seq_vals[-1]/seq_vals[0]:.2f})")

    # 2. Group by (mode, procs, threads)
    groups = defaultdict(list)
    for r in rows:
        mode = r['mode']
        p = int(r['procs'])
        t = int(r['threads'])
        rep = int(r['rep'])
        cs = float(r['compute_s'])
        ts = float(r['total_s'])
        paired_sp = seq_compute[rep] / cs
        paired_total_sp = seq_total[rep] / ts
        groups[(mode, p, t)].append({
            'compute_s': cs,
            'total_s': ts,
            'paired_sp': paired_sp,
            'paired_total_sp': paired_total_sp,
            'scatter_s': float(r['scatter_s']) if r['scatter_s'] else 0.0,
            'halo_s': float(r['halo_s']) if r['halo_s'] else 0.0,
            'conv_s': float(r['conv_s']) if r['conv_s'] else 0.0,
            'gather_s': float(r['gather_s']) if r['gather_s'] else 0.0,
        })

    # Summary table
    summary_rows = []
    for (mode, p, t), entries in groups.items():
        cs_list = sorted([e['compute_s'] for e in entries])
        ts_list = sorted([e['total_s'] for e in entries])
        sp_list = sorted([e['paired_sp'] for e in entries])
        tot_sp_list = sorted([e['paired_total_sp'] for e in entries])

        n = len(entries)
        med_cs = cs_list[n // 2]
        min_cs = cs_list[0]
        max_cs = cs_list[-1]

        med_ts = ts_list[n // 2]
        min_ts = ts_list[0]
        max_ts = ts_list[-1]

        med_sp = sp_list[n // 2]
        min_sp = sp_list[0]
        max_sp = sp_list[-1]

        med_tot_sp = tot_sp_list[n // 2]

        total_threads = p * t
        eff_hw = med_sp / total_threads
        eff_core = med_sp / min(total_threads, 6)

        summary_rows.append({
            'mode': mode,
            'procs': p,
            'threads': t,
            'total_threads': total_threads,
            'med_compute_s': f"{med_cs:.3f}",
            'min_compute_s': f"{min_cs:.3f}",
            'max_compute_s': f"{max_cs:.3f}",
            'med_total_s': f"{med_ts:.3f}",
            'min_total_s': f"{min_ts:.3f}",
            'max_total_s': f"{max_ts:.3f}",
            'med_paired_speedup': f"{med_sp:.3f}",
            'min_paired_speedup': f"{min_sp:.3f}",
            'max_paired_speedup': f"{max_sp:.3f}",
            'eff_hw_thread': f"{eff_hw:.3f}",
            'eff_core': f"{eff_core:.3f}",
            'med_total_speedup': f"{med_tot_sp:.3f}"
        })

    # Sort logically
    mode_order = {'seq': 0, 'omp': 1, 'mpi': 2, 'hybrid': 3}
    summary_rows.sort(key=lambda r: (mode_order[r['mode']], r['total_threads'], r['procs']))

    with open('results/final_summary_main.csv', 'w', newline='') as f:
        fieldnames = ['mode', 'procs', 'threads', 'total_threads',
                      'med_compute_s', 'min_compute_s', 'max_compute_s',
                      'med_total_s', 'min_total_s', 'max_total_s',
                      'med_paired_speedup', 'min_paired_speedup', 'max_paired_speedup',
                      'eff_hw_thread', 'eff_core', 'med_total_speedup']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print("Wrote results/final_summary_main.csv")

    # 3. Breakdown table for MPI-only and hybrid (3,4)
    breakdown_configs = [
        ('mpi', 1, 1),
        ('mpi', 2, 1),
        ('mpi', 4, 1),
        ('mpi', 6, 1),
        ('mpi', 12, 1),
        ('hybrid', 3, 4)
    ]
    bd_rows = []
    for cfg in breakdown_configs:
        entries = groups.get(cfg, [])
        if not entries:
            continue
        n = len(entries)
        sc = sorted([e['scatter_s'] for e in entries])[n // 2]
        hl = sorted([e['halo_s'] for e in entries])[n // 2]
        cv = sorted([e['conv_s'] for e in entries])[n // 2]
        cp = sorted([e['compute_s'] for e in entries])[n // 2]
        gt = sorted([e['gather_s'] for e in entries])[n // 2]
        tt = sorted([e['total_s'] for e in entries])[n // 2]
        non_compute_share = (tt - cp) / tt * 100.0

        bd_rows.append({
            'mode': cfg[0],
            'procs': cfg[1],
            'threads': cfg[2],
            'med_scatter_s': f"{sc:.3f}",
            'med_halo_s': f"{hl:.3f}",
            'med_conv_s': f"{cv:.3f}",
            'med_compute_s': f"{cp:.3f}",
            'med_gather_s': f"{gt:.3f}",
            'med_total_s': f"{tt:.3f}",
            'non_compute_pct': f"{non_compute_share:.1f}%"
        })

    with open('results/final_breakdown.csv', 'w', newline='') as f:
        fieldnames = ['mode', 'procs', 'threads', 'med_scatter_s', 'med_halo_s', 'med_conv_s',
                      'med_compute_s', 'med_gather_s', 'med_total_s', 'non_compute_pct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bd_rows)
    print("Wrote results/final_breakdown.csv")

def analyze_kernels():
    rows = []
    with open('results/final_kernels.csv', 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Map sequential per rep, filter, k
    seq_map = {}
    for r in rows:
        if r['mode'] == 'seq':
            k_key = (r['filter'], int(r['k']), int(r['rep']))
            seq_map[k_key] = float(r['compute_s'])

    # Group hybrid (3,4) by filter, k
    groups = defaultdict(list)
    seq_groups = defaultdict(list)
    for r in rows:
        filt = r['filter']
        k = int(r['k'])
        rep = int(r['rep'])
        if r['mode'] == 'seq':
            seq_groups[(filt, k)].append(float(r['compute_s']))
        elif r['mode'] == 'hybrid':
            cs = float(r['compute_s'])
            seq_cs = seq_map[(filt, k, rep)]
            sp = seq_cs / cs
            groups[(filt, k)].append({
                'hybrid_cs': cs,
                'paired_sp': sp
            })

    kernel_summary = []
    for (filt, k), entries in sorted(groups.items()):
        seq_vals = sorted(seq_groups[(filt, k)])
        hyb_vals = sorted([e['hybrid_cs'] for e in entries])
        sp_vals = sorted([e['paired_sp'] for e in entries])

        n = len(entries)
        med_seq = seq_vals[len(seq_vals) // 2]
        med_hyb = hyb_vals[n // 2]
        med_sp = sp_vals[n // 2]
        min_sp = sp_vals[0]
        max_sp = sp_vals[-1]

        kernel_summary.append({
            'filter': filt,
            'k': k,
            'med_seq_compute_s': f"{med_seq:.3f}",
            'med_hybrid_compute_s': f"{med_hyb:.3f}",
            'med_paired_speedup': f"{med_sp:.3f}",
            'min_paired_speedup': f"{min_sp:.3f}",
            'max_paired_speedup': f"{max_sp:.3f}"
        })

    with open('results/final_summary_kernels.csv', 'w', newline='') as f:
        fieldnames = ['filter', 'k', 'med_seq_compute_s', 'med_hybrid_compute_s',
                      'med_paired_speedup', 'min_paired_speedup', 'max_paired_speedup']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kernel_summary)
    print("Wrote results/final_summary_kernels.csv")

if __name__ == '__main__':
    analyze_main()
    analyze_kernels()

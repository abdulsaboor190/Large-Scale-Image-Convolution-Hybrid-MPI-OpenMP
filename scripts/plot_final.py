import csv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11})

def read_csv_dicts(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def plot_speedup():
    rows = read_csv_dicts('results/final_summary_main.csv')
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Ideal linear line
    x_ideal = np.array([1, 2, 3, 4, 6, 8, 12])
    ax.plot(x_ideal, x_ideal, 'k--', label='Ideal linear speedup', alpha=0.6)

    # Filter by mode
    omp_rows = sorted([r for r in rows if r['mode'] == 'omp'], key=lambda x: int(x['total_threads']))
    mpi_rows = sorted([r for r in rows if r['mode'] == 'mpi'], key=lambda x: int(x['total_threads']))
    hyb_rows = [r for r in rows if r['mode'] == 'hybrid']

    # OpenMP
    x_omp = [int(r['total_threads']) for r in omp_rows]
    y_omp = [float(r['med_paired_speedup']) for r in omp_rows]
    yerr_omp = [[float(r['med_paired_speedup']) - float(r['min_paired_speedup']) for r in omp_rows],
                [float(r['max_paired_speedup']) - float(r['med_paired_speedup']) for r in omp_rows]]
    ax.errorbar(x_omp, y_omp, yerr=yerr_omp, marker='o', capsize=4,
                label='OpenMP only (1 process, T threads)', color='#1f77b4', lw=2)

    # MPI
    x_mpi = [int(r['total_threads']) for r in mpi_rows]
    y_mpi = [float(r['med_paired_speedup']) for r in mpi_rows]
    yerr_mpi = [[float(r['med_paired_speedup']) - float(r['min_paired_speedup']) for r in mpi_rows],
                [float(r['max_paired_speedup']) - float(r['med_paired_speedup']) for r in mpi_rows]]
    ax.errorbar(x_mpi, y_mpi, yerr=yerr_mpi, marker='s', capsize=4,
                label='MPI only (P processes, 1 thread)', color='#2ca02c', lw=2)

    # Hybrid
    for r in hyb_rows:
        x = int(r['total_threads'])
        y = float(r['med_paired_speedup'])
        err = [[y - float(r['min_paired_speedup'])], [float(r['max_paired_speedup']) - y]]
        ax.errorbar([x], [y], yerr=err, marker='^', capsize=4, color='#d62728', markersize=8)
        ax.annotate(f"{r['procs']}x{r['threads']}", (x, y), textcoords="offset points", xytext=(-10, 8), fontsize=9)

    ax.plot([], [], marker='^', color='#d62728', linestyle='None', label='Hybrid (P x T)')

    ax.axvline(x=6, color='grey', linestyle=':', lw=1.5)
    ax.text(6.1, 1.5, '6 Physical Cores (SMT beyond)', color='#555555', rotation=90, fontsize=10)

    ax.set_xlabel('Total Parallel Threads (P x T)')
    ax.set_ylabel('Paired Speedup (T_seq / T_compute)')
    ax.set_title('Gaussian 5x5 Speedup vs Total Threads on 20,000x20,000 PGM')
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12])
    ax.set_ylim(0, 13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('results/final_speedup.png')
    plt.close()
    print("Saved results/final_speedup.png")

def plot_efficiency():
    rows = read_csv_dicts('results/final_summary_main.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    omp_rows = sorted([r for r in rows if r['mode'] == 'omp'], key=lambda x: int(x['total_threads']))
    mpi_rows = sorted([r for r in rows if r['mode'] == 'mpi'], key=lambda x: int(x['total_threads']))
    hyb_rows = [r for r in rows if r['mode'] == 'hybrid']

    # 1. Hardware Thread Efficiency
    ax1.plot([int(r['total_threads']) for r in omp_rows],
             [float(r['eff_hw_thread']) for r in omp_rows],
             marker='o', color='#1f77b4', lw=2, label='OpenMP only')
    ax1.plot([int(r['total_threads']) for r in mpi_rows],
             [float(r['eff_hw_thread']) for r in mpi_rows],
             marker='s', color='#2ca02c', lw=2, label='MPI only')

    for r in hyb_rows:
        x = int(r['total_threads'])
        y = float(r['eff_hw_thread'])
        ax1.scatter([x], [y], marker='^', color='#d62728', s=60, zorder=5)
        ax1.annotate(f"{r['procs']}x{r['threads']}", (x, y), textcoords="offset points", xytext=(-10, 6), fontsize=8)

    ax1.scatter([], [], marker='^', color='#d62728', label='Hybrid')
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=6, color='grey', linestyle=':', lw=1.2)
    ax1.set_xlabel('Total Hardware Threads (P x T)')
    ax1.set_ylabel('Efficiency per Hardware Thread')
    ax1.set_title('Efficiency per Hardware Thread (P x T)')
    ax1.set_xticks([1, 2, 3, 4, 6, 8, 12])
    ax1.set_ylim(0, 1.3)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # 2. Physical Core Efficiency
    ax2.plot([int(r['total_threads']) for r in omp_rows],
             [float(r['eff_core']) for r in omp_rows],
             marker='o', color='#1f77b4', lw=2, label='OpenMP only')
    ax2.plot([int(r['total_threads']) for r in mpi_rows],
             [float(r['eff_core']) for r in mpi_rows],
             marker='s', color='#2ca02c', lw=2, label='MPI only')

    for r in hyb_rows:
        x = int(r['total_threads'])
        y = float(r['eff_core'])
        ax2.scatter([x], [y], marker='^', color='#d62728', s=60, zorder=5)
        ax2.annotate(f"{r['procs']}x{r['threads']}", (x, y), textcoords="offset points", xytext=(-10, 6), fontsize=8)

    ax2.scatter([], [], marker='^', color='#d62728', label='Hybrid')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=6, color='grey', linestyle=':', lw=1.2)
    ax2.set_xlabel('Total Threads (P x T)')
    ax2.set_ylabel('Efficiency per Physical Core (min(P x T, 6))')
    ax2.set_title('Efficiency per Physical Core (Capped at 6)')
    ax2.set_xticks([1, 2, 3, 4, 6, 8, 12])
    ax2.set_ylim(0, 1.4)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/final_efficiency.png')
    plt.close()
    print("Saved results/final_efficiency.png")

def plot_hybrid_compare():
    rows = read_csv_dicts('results/final_summary_main.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    row_map = {(r['mode'], int(r['procs']), int(r['threads'])): r for r in rows}

    combos_12 = [
        ('omp', 1, 12, 'OMP 1x12'),
        ('hybrid', 2, 6, 'Hyb 2x6'),
        ('hybrid', 3, 4, 'Hyb 3x4'),
        ('hybrid', 4, 3, 'Hyb 4x3'),
        ('hybrid', 6, 2, 'Hyb 6x2'),
        ('mpi', 12, 1, 'MPI 12x1')
    ]
    labels_12, sp_12, err_l_12, err_h_12 = [], [], [], []
    for m, p, t, l in combos_12:
        r = row_map[(m, p, t)]
        labels_12.append(l)
        sp = float(r['med_paired_speedup'])
        sp_12.append(sp)
        err_l_12.append(sp - float(r['min_paired_speedup']))
        err_h_12.append(float(r['max_paired_speedup']) - sp)

    bars1 = ax1.bar(labels_12, sp_12, yerr=[err_l_12, err_h_12], capsize=5,
                    color=['#1f77b4', '#d62728', '#2ca02c', '#d62728', '#d62728', '#9467bd'], alpha=0.85)
    ax1.set_ylabel('Paired Speedup (T_seq / T_compute)')
    ax1.set_title('All 12-Thread Configurations (P x T = 12)')
    ax1.set_ylim(0, 8.0)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f}x", ha='center', va='bottom', fontsize=9)

    combos_6 = [
        ('omp', 1, 6, 'OMP 1x6'),
        ('hybrid', 2, 3, 'Hyb 2x3'),
        ('hybrid', 3, 2, 'Hyb 3x2'),
        ('mpi', 6, 1, 'MPI 6x1')
    ]
    labels_6, sp_6, err_l_6, err_h_6 = [], [], [], []
    for m, p, t, l in combos_6:
        r = row_map[(m, p, t)]
        labels_6.append(l)
        sp = float(r['med_paired_speedup'])
        sp_6.append(sp)
        err_l_6.append(sp - float(r['min_paired_speedup']))
        err_h_6.append(float(r['max_paired_speedup']) - sp)

    bars2 = ax2.bar(labels_6, sp_6, yerr=[err_l_6, err_h_6], capsize=5,
                    color=['#1f77b4', '#d62728', '#d62728', '#9467bd'], alpha=0.85)
    ax2.set_ylabel('Paired Speedup (T_seq / T_compute)')
    ax2.set_title('All 6-Physical-Core Configurations (P x T = 6)')
    ax2.set_ylim(0, 7.0)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f}x", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/final_hybrid_compare.png')
    plt.close()
    print("Saved results/final_hybrid_compare.png")

def plot_kernels():
    rows = read_csv_dicts('results/final_summary_kernels.csv')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    filters = ['gaussian', 'sharpen', 'sobel']
    colors = {'gaussian': '#2ca02c', 'sharpen': '#1f77b4', 'sobel': '#d62728'}
    markers = {'gaussian': 'o', 'sharpen': 's', 'sobel': '^'}

    for filt in filters:
        sub = sorted([r for r in rows if r['filter'] == filt], key=lambda x: int(x['k']))
        x = [int(r['k']) for r in sub]
        y = [float(r['med_paired_speedup']) for r in sub]
        yerr = [[y[i] - float(sub[i]['min_paired_speedup']) for i in range(len(sub))],
                [float(sub[i]['max_paired_speedup']) - y[i] for i in range(len(sub))]]
        ax.errorbar(x, y, yerr=yerr, marker=markers[filt], color=colors[filt], lw=2, capsize=4,
                    label=f"{filt.capitalize()} (Hybrid 3x4)")

    ax.set_xlabel('Kernel Size (k x k)')
    ax.set_ylabel('Paired Speedup (T_seq / T_compute)')
    ax.set_title('Hybrid (3 ranks x 4 threads) Speedup by Kernel Size')
    ax.set_xticks([3, 5, 7])
    ax.set_xticklabels(['3x3', '5x5', '7x7'])
    ax.set_ylim(4.0, 7.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('results/final_kernels.png')
    plt.close()
    print("Saved results/final_kernels.png")

if __name__ == '__main__':
    plot_speedup()
    plot_efficiency()
    plot_hybrid_compare()
    plot_kernels()

# Phase 3 Design: Pure MPI 1-D Row-Block Decomposition

## 1. Domain Decomposition & Active Ranks Rule

### Concept
The 2D image of dimension $W \times H$ is decomposed along the vertical (row) dimension (1-D block decomposition).
Each convolution filter has kernel size $k$ and radius $r = \lfloor k / 2 \rfloor$.
To compute valid convolution on a local block of rows without inter-rank dependencies during computation, each process requires a **halo (ghost cell)** of $r$ rows above and $r$ rows below its local block.

### Active Rank Rule
If a process owns fewer rows than the halo radius ($rows < r$), halo exchange cannot be safely done in a simple single exchange phase between immediate neighbors because halos would span across multiple processes.
To avoid multi-hop communication and maintain a simple, robust design:
$$P_A = \max(1, \min(P, \lfloor H / r \rfloor))$$
- **Active ranks ($0 \le \text{rank} < P_A$):** partition the $H$ rows contiguously.
  - Base rows per active rank: $base = \lfloor H / P_A \rfloor$
  - Remainder: $rem = H \pmod{P_A}$
  - Rank $i < rem$ gets $base + 1$ rows; rank $i \ge rem$ gets $base$ rows.
  - Because $P_A \le \lfloor H / r \rfloor$, every active rank has at least $r$ rows (guaranteed whenever $P_A \ge 2$).
- **Idle ranks ($P_A \le \text{rank} < P$):** own 0 rows ($rows = 0$).
  - Idle ranks still participate in all collectives (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Gatherv`, `MPI_Reduce`, `MPI_Barrier`) with send/receive counts of 0 to prevent deadlock.

### Worked Example ($H = 10, P = 4, k = 5 \implies r = 2$)
- $P_A = \max(1, \min(4, \lfloor 10 / 2 \rfloor)) = \min(4, 5) = 4$. All 4 ranks active.
- $base = \lfloor 10 / 4 \rfloor = 2$, $rem = 10 \pmod 4 = 2$.
- **Rank 0:** owns rows $[0, 2]$ (count = 3), offset = 0.
- **Rank 1:** owns rows $[3, 5]$ (count = 3), offset = 3.
- **Rank 2:** owns rows $[6, 7]$ (count = 2), offset = 6.
- **Rank 3:** owns rows $[8, 9]$ (count = 2), offset = 8.
- Notice every rank owns $\ge 2$ rows, which is $\ge r$.

---

## 2. Local Buffer Layout & Halo Exchange

### Local Memory Layout
Each active rank allocates a contiguous input buffer of $(rows + 2r) \times W$ bytes (`uint8_t`):
- `[0, r)`: Top halo ($r \times W$ bytes)
- `[r, r + rows)`: Local data ($rows \times W$ bytes, received from rank 0 via `MPI_Scatterv`)
- `[r + rows, rows + 2r)`: Bottom halo ($r \times W$ bytes)

The output buffer allocated by each active rank is strictly $rows \times W$ bytes (no halos needed on output).

### Halo Exchange via Blocking `MPI_Sendrecv`
Only active ranks ($rank < P_A$) participate in the point-to-point halo exchange. Missing neighbors at global borders use `MPI_PROC_NULL` (which acts as a no-op in MPI point-to-point operations):
- `top_neighbor` = $(rank > 0) \;?\; (rank - 1) : \text{MPI\_PROC\_NULL}$
- `bottom_neighbor` = $(rank < P_A - 1) \;?\; (rank + 1) : \text{MPI\_PROC\_NULL}$

Exchange steps:
1. **Exchange 1:** Send first $r$ local rows to `top_neighbor`, receive into bottom halo from `bottom_neighbor`:
   - Send buffer: `local_buf + r * W`, count: $r \times W$, dest: `top_neighbor`, tag: 100
   - Recv buffer: `local_buf + (r + rows) * W`, count: $r \times W$, source: `bottom_neighbor`, tag: 100
2. **Exchange 2:** Send last $r$ local rows to `bottom_neighbor`, receive into top halo from `top_neighbor`:
   - Send buffer: `local_buf + rows * W`, count: $r \times W$, dest: `bottom_neighbor`, tag: 200
   - Recv buffer: `local_buf`, count: $r \times W$, source: `top_neighbor`, tag: 200

### Global Boundary Handling (Clamp-to-Edge)
In accordance with `docs/DECISIONS.md`:
- If `rank == 0` (`top_neighbor == MPI_PROC_NULL`):
  Top halo (rows $0 \dots r-1$) is filled with $r$ copies of the first local row (`local_buf + r * W`).
- If `rank == P_A - 1` (`bottom_neighbor == MPI_PROC_NULL`):
  Bottom halo (rows $r + rows \dots rows + 2r - 1$) is filled with $r$ copies of the last local row (`local_buf + (r + rows - 1) * W`).

---

## 3. Computation & Shared Kernel Reuse

Each active rank invokes the existing shared convolution function from `src/conv_core.h`:
- Treat `local_buf` as an image of width $W$, total height $H_{local} = rows + 2r$.
- Compute row range: $y_{start} = r$ to $y_{end} = r + rows$.
- Output buffer has size $rows \times W$ and is indexed relative to local row 0:
  `out_buf[(y - r) * W + x]`.

Because the local buffer already contains exact clamped halo rows for $y < r$ and $y \ge r + rows$, inside the range $[r, r + rows)$ the y-coordinate clamping in `conv_core.h` is never triggered; it accesses pre-exchanged or boundary-replicated rows directly, guaranteeing **bit-exact identity** with sequential execution.

---

## 4. Collective Communications & Rank 0 Flow

1. **Header broadcast:** Rank 0 reads PGM header; broadcasts $(W, H, \text{status})$ using `MPI_Bcast`.
   - If header read fails, broadcast error flag, call `MPI_Finalize()`, and exit with non-zero code across all ranks (no deadlock).
2. **Distribution:** Rank 0 reads the full input image ($W \times H$).
   - `MPI_Scatterv`: send counts and displacements computed on rank 0.
   - Active ranks receive into `local_buf + r * W`. Idle ranks receive 0 bytes.
   - Immediately after scatter, rank 0 frees the full input buffer to conserve memory.
3. **Gathering:**
   - Active ranks compute into `local_out`.
   - Rank 0 allocates full output buffer ($W \times H$).
   - `MPI_Gatherv`: collects $rows \times W$ from each active rank into rank 0's output buffer.
4. **Writing:** Rank 0 writes `out.pgm`.

---

## 5. Memory Footprint on 20,000 × 20,000 Image

- Image size: $20000 \times 20000 \times 1\text{ byte} = 400\text{ MB}$.
- **Rank 0 Peak Memory:**
  - Before scatter: Full input buffer (400 MB).
  - Local slice: e.g. for $P=12$, $(1667 + 2 \times 3) \times 20000 \approx 33.5\text{ MB}$.
  - Rank 0 frees full input image immediately after `MPI_Scatterv`.
  - At gather: Full output buffer (400 MB) is allocated right before `MPI_Gatherv`.
  - Peak memory on Rank 0 $\le 450\text{ MB}$, well within WSL's available ~3.5 to 5 GB RAM.
- **Worker Ranks Peak Memory:**
  - Local input with halo + local output: $\approx 33.5\text{ MB} + 33.3\text{ MB} \approx 67\text{ MB}$ total.

---

## 6. Timing Protocol & Metrics

Timing is measured using `MPI_Wtime()`, synchronized across all ranks with `MPI_Barrier()` before each timed region, and reduced to rank 0 using `MPI_Reduce(..., MPI_MAX)`:
- `scatter_s`: `MPI_Scatterv` distribution time.
- `halo_s`: Halo exchange + border replication time.
- `conv_s`: Core convolution computation time.
- `compute_s`: `halo_s + conv_s` (parallel compute region, max across all ranks).
- `gather_s`: `MPI_Gatherv` collection time.
- `total_s`: Full wall-clock time from start to finish (including disk I/O).

Output format on stdout (rank 0 only, everything else to stderr):
```
RESULT mode=mpi filter=<filter> k=<k> w=<w> h=<h> procs=<P> active=<P_A> threads=1 scatter_s=.. halo_s=.. conv_s=.. compute_s=.. gather_s=.. total_s=..
```

---

## 7. `mpirun` Binding Recipe for Ryzen 5 7430U (6 Cores / 12 SMT Threads)

- For $P \le 6$:
  `mpirun --bind-to core --map-by core -np P ./build/mpi_convolve ...`
  (Binds 1 rank per physical core without oversubscription).
- For $P > 6$ (up to $P = 12$):
  `mpirun --use-hwthread-cpus --bind-to hwthread --map-by hwthread -np P ./build/mpi_convolve ...`
  (Allows mapping across all 12 hardware SMT threads without oversubscription warnings).
---

## 8. Quick Timing Sanity Results (Gaussian 5x5, 20,000x20,000 Image)

Measured on AMD Ryzen 5 7430U (6 physical cores, 12 SMT threads).
Sequential baseline `compute_s` = 11.488 s (from `results/baseline_seq_summary.csv`).

| P | active | scatter_s | halo_s | conv_s | compute_s | gather_s | total_s | Speedup ($T_{seq}/T_{comp}$) | Efficiency |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.367 s | 0.000 s | 10.432 s | 10.432 s | 0.273 s | 12.375 s | 1.101 | 1.101 |
| 2 | 2 | 0.344 s | 0.000 s | 5.963 s | 5.964 s | 0.264 s | 7.176 s | 1.926 | 0.963 |
| 4 | 4 | 0.365 s | 0.000 s | 3.629 s | 3.629 s | 0.268 s | 5.033 s | 3.166 | 0.791 |
| 6 | 6 | 0.436 s | 0.000 s | 3.269 s | 3.269 s | 0.274 s | 4.902 s | 3.514 | 0.586 |
| 12 | 12 | 0.548 s | 0.001 s | 2.178 s | 2.178 s | 0.337 s | 4.243 s | 5.275 | 0.440 |

**Notes:**
- At $P = 1$, `compute_s` = 10.432 s vs sequential baseline 11.488 s (within 9.2%, meeting the 10% target).
- Halo exchange time is sub-millisecond ($< 0.001$ s) because each rank exchanges only $2 \times 20,000 = 40,000$ bytes locally over RAM.
- Speedup scales up to 5.28x on 12 SMT threads.

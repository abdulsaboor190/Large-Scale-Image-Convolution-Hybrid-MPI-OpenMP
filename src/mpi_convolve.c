/*
 * mpi_convolve.c - Hybrid MPI + OpenMP image convolution with 1-D row-block decomposition.
 *
 * Usage:
 *   mpirun -np P ./build/mpi_convolve <in.pgm> <out.pgm> <sobel|gaussian|sharpen> <3|5|7> [threads]
 *
 * Phase 4 hybrid design rules (docs/hybrid_design.md):
 *   - MPI_Init_thread with MPI_THREAD_FUNNELED (abort if provided < FUNNELED).
 *   - Optional 5th argument [threads] (default: 1, range: 1 to 64).
 *   - If threads == 1, mode=mpi; if threads > 1, mode=hybrid.
 *   - All MPI calls, scatter/gather, halo exchange and I/O are performed by the master thread.
 *   - Inside each active rank, local rows [radius, radius + local_rows) are parallelized with
 *     OpenMP using schedule(runtime) and locked default omp_sched_static, chunk 0.
 *   - Arithmetic reuses conv_core.h (convolve_rows, sobel_rows) for bit-exact identity.
 *   - stdout (rank 0 only):
 *       RESULT mode=<mpi|hybrid> filter=... k=... w=... h=... procs=... active=... threads=... \
 *              scatter_s=... halo_s=... conv_s=... compute_s=... gather_s=... total_s=...
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conv_core.h"
#include "pgm_io.h"

#define CHOSEN_SCHED_KIND  omp_sched_static
#define CHOSEN_SCHED_CHUNK 0

static void install_schedule(void)
{
    const char *env = getenv("OMP_SCHEDULE");
    if (env != NULL && env[0] != '\0')
        return;
    omp_set_schedule(CHOSEN_SCHED_KIND, CHOSEN_SCHED_CHUNK);
}

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s <in.pgm> <out.pgm> <sobel|gaussian|sharpen> <3|5|7> [threads]\n", prog);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "error: MPI does not provide MPI_THREAD_FUNNELED (provided=%d)\n", provided);
        }
        MPI_Finalize();
        return 1;
    }

    double t_start = MPI_Wtime();

    if (argc != 5 && argc != 6) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *in_path = argv[1];
    const char *out_path = argv[2];
    const char *filter = argv[3];

    char *end = NULL;
    errno = 0;
    long ksize_long = strtol(argv[4], &end, 10);
    if (errno != 0 || end == argv[4] || *end != '\0' ||
        (ksize_long != 3 && ksize_long != 5 && ksize_long != 7)) {
        if (rank == 0) fprintf(stderr, "kernel size must be 3, 5, or 7 (got '%s')\n", argv[4]);
        MPI_Finalize();
        return 1;
    }
    int k = (int)ksize_long;

    int threads = 1;
    if (argc == 6) {
        errno = 0;
        char *tend = NULL;
        long t_long = strtol(argv[5], &tend, 10);
        if (errno != 0 || tend == argv[5] || *tend != '\0' || t_long < 1 || t_long > 64) {
            if (rank == 0) fprintf(stderr, "threads must be between 1 and 64 (got '%s')\n", argv[5]);
            MPI_Finalize();
            return 1;
        }
        threads = (int)t_long;
    }

    /* Set OpenMP thread count and schedule */
    omp_set_num_threads(threads);
    install_schedule();

    int is_sobel = (strcmp(filter, "sobel") == 0);
    int is_gaussian = (strcmp(filter, "gaussian") == 0);
    int is_sharpen = (strcmp(filter, "sharpen") == 0);

    if (!is_sobel && !is_gaussian && !is_sharpen) {
        if (rank == 0) fprintf(stderr, "error: unknown filter '%s'\n", filter);
        MPI_Finalize();
        return 1;
    }

    kernel_t k_a, k_b;
    if (is_sobel) {
        if (kernel_make(K_SOBEL_X, k, &k_a) != 0 || kernel_make(K_SOBEL_Y, k, &k_b) != 0) {
            if (rank == 0) fprintf(stderr, "failed to build Sobel kernels of size %d\n", k);
            MPI_Finalize();
            return 1;
        }
    } else {
        kernel_type_t type = is_gaussian ? K_GAUSSIAN : K_SHARPEN;
        if (kernel_make(type, k, &k_a) != 0) {
            if (rank == 0) fprintf(stderr, "failed to build %s kernel of size %d\n", filter, k);
            MPI_Finalize();
            return 1;
        }
    }

    int radius = k_a.radius;

    /*
     * Rank 0 reads the PGM header and checks for read validity.
     * Broadcast status: status_ok == 1 means valid header, 0 means failure.
     */
    int header_info[3] = {0, 0, 0}; /* width, height, status_ok */
    long data_offset = 0;
    if (rank == 0) {
        int w = 0, h = 0;
        if (pgm_read_header(in_path, &w, &h, &data_offset) != 0) {
            fprintf(stderr, "error: rank 0 failed to read PGM header from %s\n", in_path);
            header_info[2] = 0;
        } else {
            header_info[0] = w;
            header_info[1] = h;
            header_info[2] = 1;
        }
    }

    MPI_Bcast(header_info, 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (header_info[2] != 1) {
        MPI_Finalize();
        return 1;
    }

    int width = header_info[0];
    int height = header_info[1];

    /*
     * Active ranks calculation:
     * PA = max(1, min(P, height / radius))
     * If height < radius, PA = 1.
     */
    int max_active = (radius > 0) ? (height / radius) : 1;
    if (max_active < 1) max_active = 1;
    int PA = size;
    if (PA > max_active) PA = max_active;
    if (PA < 1) PA = 1;

    /*
     * Domain decomposition:
     * Distribute height rows across PA active ranks.
     * Ranks >= PA get 0 rows.
     */
    int *sendcounts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    if (!sendcounts || !displs) {
        fprintf(stderr, "rank %d: memory allocation failed for scatter counts\n", rank);
        free(sendcounts);
        free(displs);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int base_rows = height / PA;
    int rem_rows = height % PA;
    int current_offset = 0;

    for (int p = 0; p < size; p++) {
        if (p < PA) {
            int r_count = base_rows + (p < rem_rows ? 1 : 0);
            size_t bytes = (size_t)r_count * (size_t)width;
            if (bytes > (size_t)INT_MAX) {
                if (rank == 0) fprintf(stderr, "error: row count bytes exceed INT_MAX for rank %d\n", p);
                free(sendcounts);
                free(displs);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            sendcounts[p] = (int)bytes;
            displs[p] = (int)((size_t)current_offset * (size_t)width);
            current_offset += r_count;
        } else {
            sendcounts[p] = 0;
            displs[p] = 0;
        }
    }

    int local_rows = 0;
    if (rank < PA) {
        local_rows = base_rows + (rank < rem_rows ? 1 : 0);
    }

    /*
     * Rank 0 reads full image from disk before scatter.
     */
    unsigned char *full_in = NULL;
    if (rank == 0) {
        full_in = (unsigned char *)malloc((size_t)width * (size_t)height);
        if (!full_in) {
            fprintf(stderr, "error: rank 0 failed to allocate full input buffer (%zu bytes)\n",
                    (size_t)width * (size_t)height);
            header_info[2] = 0;
        } else {
            if (pgm_read_rows(in_path, data_offset, width, 0, height, full_in) != 0) {
                fprintf(stderr, "error: rank 0 failed to read PGM data from %s\n", in_path);
                header_info[2] = 0;
            } else {
                header_info[2] = 1;
            }
        }
    }

    /* Check pgm_read_rows status */
    int load_ok = header_info[2];
    MPI_Bcast(&load_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!load_ok) {
        if (full_in) free(full_in);
        free(sendcounts);
        free(displs);
        MPI_Finalize();
        return 1;
    }

    /*
     * Allocate local input buffer: (local_rows + 2 * radius) * width bytes.
     */
    size_t local_in_bytes = (size_t)(local_rows + 2 * radius) * (size_t)width;
    unsigned char *local_in = NULL;
    if (local_rows > 0) {
        local_in = (unsigned char *)malloc(local_in_bytes);
        if (!local_in) {
            fprintf(stderr, "rank %d: failed to allocate local input buffer (%zu bytes)\n", rank, local_in_bytes);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    /*
     * Scatterv Timing
     */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_scatter_start = MPI_Wtime();

    unsigned char *recv_ptr = (local_rows > 0) ? (local_in + (size_t)radius * (size_t)width) : NULL;
    int recv_count = sendcounts[rank];

    MPI_Scatterv(full_in, sendcounts, displs, MPI_UNSIGNED_CHAR,
                 recv_ptr, recv_count, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    double t_scatter_end = MPI_Wtime();
    double local_scatter_s = t_scatter_end - t_scatter_start;

    /* Rank 0 immediately frees full input buffer to conserve memory */
    if (rank == 0 && full_in) {
        free(full_in);
        full_in = NULL;
    }

    /*
     * Halo Exchange Timing (includes global boundary replication)
     */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_halo_start = MPI_Wtime();

    if (local_rows > 0) {
        int top_nbr = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
        int bot_nbr = (rank < PA - 1) ? (rank + 1) : MPI_PROC_NULL;

        size_t halo_bytes = (size_t)radius * (size_t)width;
        if (halo_bytes > (size_t)INT_MAX) {
            fprintf(stderr, "rank %d: halo size exceeds INT_MAX\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        int halo_count = (int)halo_bytes;

        /* Step 1: Send top rows to top_nbr, receive into bottom halo from bot_nbr */
        unsigned char *send_top = local_in + (size_t)radius * (size_t)width;
        unsigned char *recv_bot = local_in + (size_t)(radius + local_rows) * (size_t)width;

        MPI_Sendrecv(send_top, halo_count, MPI_UNSIGNED_CHAR, top_nbr, 100,
                     recv_bot, halo_count, MPI_UNSIGNED_CHAR, bot_nbr, 100,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Step 2: Send bottom rows to bot_nbr, receive into top halo from top_nbr */
        unsigned char *send_bot = local_in + (size_t)local_rows * (size_t)width;
        unsigned char *recv_top = local_in;

        MPI_Sendrecv(send_bot, halo_count, MPI_UNSIGNED_CHAR, bot_nbr, 200,
                     recv_top, halo_count, MPI_UNSIGNED_CHAR, top_nbr, 200,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Global boundary clamping */
        if (top_nbr == MPI_PROC_NULL) {
            const unsigned char *row0 = local_in + (size_t)radius * (size_t)width;
            for (int r = 0; r < radius; r++) {
                memcpy(local_in + (size_t)r * (size_t)width, row0, (size_t)width);
            }
        }

        if (bot_nbr == MPI_PROC_NULL) {
            const unsigned char *row_last = local_in + (size_t)(radius + local_rows - 1) * (size_t)width;
            for (int r = 0; r < radius; r++) {
                memcpy(local_in + (size_t)(radius + local_rows + r) * (size_t)width, row_last, (size_t)width);
            }
        }
    }

    double t_halo_end = MPI_Wtime();
    double local_halo_s = t_halo_end - t_halo_start;

    /*
     * Convolution Computation Timing (Parallelized with OpenMP threads inside each rank)
     */
    size_t local_out_bytes = (size_t)local_rows * (size_t)width;
    unsigned char *local_out = NULL;
    if (local_rows > 0) {
        local_out = (unsigned char *)malloc(local_out_bytes);
        if (!local_out) {
            fprintf(stderr, "rank %d: failed to allocate local output buffer (%zu bytes)\n", rank, local_out_bytes);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_conv_start = MPI_Wtime();

    if (local_rows > 0) {
        int total_h = local_rows + 2 * radius;
        size_t total_out_bytes = (size_t)total_h * (size_t)width;
        unsigned char *conv_out_buf = (unsigned char *)malloc(total_out_bytes);
        if (!conv_out_buf) {
            fprintf(stderr, "rank %d: failed to allocate conv_out_buf\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        if (threads > 1) {
            /* OpenMP multi-threaded row computation */
            if (is_sobel) {
                #pragma omp parallel for schedule(runtime)
                for (int y = radius; y < radius + local_rows; y++) {
                    sobel_rows(local_in, conv_out_buf, width, total_h, y, y + 1, &k_a, &k_b);
                }
            } else {
                #pragma omp parallel for schedule(runtime)
                for (int y = radius; y < radius + local_rows; y++) {
                    convolve_rows(local_in, conv_out_buf, width, total_h, y, y + 1, &k_a);
                }
            }
        } else {
            /* Pure sequential invocation inside rank */
            if (is_sobel) {
                sobel_rows(local_in, conv_out_buf, width, total_h, radius, radius + local_rows, &k_a, &k_b);
            } else {
                convolve_rows(local_in, conv_out_buf, width, total_h, radius, radius + local_rows, &k_a);
            }
        }

        memcpy(local_out, conv_out_buf + (size_t)radius * (size_t)width, local_out_bytes);
        free(conv_out_buf);
    }

    double t_conv_end = MPI_Wtime();
    double local_conv_s = t_conv_end - t_conv_start;

    /* Free local input buffer */
    if (local_in) {
        free(local_in);
        local_in = NULL;
    }

    /*
     * Gather Timing
     */
    unsigned char *full_out = NULL;
    if (rank == 0) {
        full_out = (unsigned char *)malloc((size_t)width * (size_t)height);
        if (!full_out) {
            fprintf(stderr, "error: rank 0 failed to allocate full output buffer (%zu bytes)\n",
                    (size_t)width * (size_t)height);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_gather_start = MPI_Wtime();

    MPI_Gatherv(local_out, recv_count, MPI_UNSIGNED_CHAR,
                full_out, sendcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    double t_gather_end = MPI_Wtime();
    double local_gather_s = t_gather_end - t_gather_start;

    if (local_out) {
        free(local_out);
        local_out = NULL;
    }

    /*
     * Reduce timings across ranks: max over ranks
     */
    double max_scatter_s = 0.0, max_halo_s = 0.0, max_conv_s = 0.0;
    double max_compute_s = 0.0, max_gather_s = 0.0;
    double local_compute_s = local_halo_s + local_conv_s;

    MPI_Reduce(&local_scatter_s, &max_scatter_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_halo_s, &max_halo_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_conv_s, &max_conv_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_s, &max_compute_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_gather_s, &max_gather_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /*
     * Rank 0 writes output PGM and prints RESULT line
     */
    int write_rc = 0;
    if (rank == 0) {
        FILE *fp = fopen(out_path, "wb");
        if (!fp) {
            fprintf(stderr, "cannot open '%s' for writing: %s\n", out_path, strerror(errno));
            write_rc = 1;
        } else {
            long out_offset = 0;
            if (pgm_write_header(fp, width, height, &out_offset) != 0 ||
                pgm_write_rows(fp, out_offset, width, 0, height, full_out) != 0) {
                fprintf(stderr, "failed writing PGM image to '%s'\n", out_path);
                write_rc = 1;
            }
            if (fclose(fp) != 0) {
                fprintf(stderr, "failed to close '%s': %s\n", out_path, strerror(errno));
                write_rc = 1;
            }
        }
        free(full_out);
        full_out = NULL;
    }

    free(sendcounts);
    free(displs);

    MPI_Bcast(&write_rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (write_rc != 0) {
        MPI_Finalize();
        return 1;
    }

    double t_end = MPI_Wtime();
    double total_s = t_end - t_start;

    if (rank == 0) {
        const char *mode = (threads == 1) ? "mpi" : "hybrid";
        printf("RESULT mode=%s filter=%s k=%d w=%d h=%d procs=%d active=%d threads=%d "
               "scatter_s=%.3f halo_s=%.3f conv_s=%.3f compute_s=%.3f gather_s=%.3f total_s=%.3f\n",
               mode, filter, k, width, height, size, PA, threads,
               max_scatter_s, max_halo_s, max_conv_s, max_compute_s, max_gather_s, total_s);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}

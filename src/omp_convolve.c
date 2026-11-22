/*
 * omp_convolve.c - one-node OpenMP convolution.
 *
 * Usage:
 *   ./build/omp_convolve <in.pgm> <out.pgm> <sobel|gaussian|sharpen> <3|5|7> [threads]
 *
 * Same pixels as seq_convolve. The only change is that output rows are split
 * across threads. Each pixel still goes through convolve_rows or sobel_rows
 * in conv_core.h, so the PGM is bit-identical to the sequential program.
 *
 * [threads] is passed to omp_set_num_threads. If it is omitted, the OpenMP
 * runtime uses OMP_NUM_THREADS (its own default when that is unset).
 *
 * stdout is exactly one line:
 *   RESULT filter=sobel k=3 w=20000 h=20000 threads=6 schedule=static compute_s=2.345 total_s=3.210
 * Everything else goes to stderr.
 * compute_s is the parallel loop only. total_s includes reading and writing.
 */
#include <errno.h>
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conv_core.h"
#include "pgm_io.h"

/*
 * Schedule chosen from results/omp_schedule_tuning.csv.
 * See docs/omp_tuning.md. static with chunk 0 had the lowest sum of
 * median compute times. Chunk 0 tells OpenMP to pick the chunk: for
 * static that is one contiguous block of about height/threads rows
 * per thread.
 *
 * The loop uses schedule(runtime), so this value is what actually runs.
 * Set OMP_SCHEDULE (for example OMP_SCHEDULE=dynamic,16) to try something
 * else without editing this file. When that variable is set, we leave the
 * runtime's value alone.
 */
#define CHOSEN_SCHED_KIND  omp_sched_static
#define CHOSEN_SCHED_CHUNK 0

static void install_schedule(void)
{
    const char *env = getenv("OMP_SCHEDULE");
    if (env != NULL && env[0] != '\0')
        return;
    omp_set_schedule(CHOSEN_SCHED_KIND, CHOSEN_SCHED_CHUNK);
}

static const char *schedule_name(omp_sched_t kind)
{
    /*
     * libgomp ORs omp_sched_monotonic into the kind it returns.
     * That bit only says chunks are handed out in order. The name on the
     * RESULT line is the schedule itself: static, dynamic, guided, or auto.
     */
    unsigned base = (unsigned)kind & ~(unsigned)omp_sched_monotonic;
    if (base == (unsigned)omp_sched_static) return "static";
    if (base == (unsigned)omp_sched_dynamic) return "dynamic";
    if (base == (unsigned)omp_sched_guided) return "guided";
    if (base == (unsigned)omp_sched_auto) return "auto";
    return "other";
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s <in.pgm> <out.pgm> <sobel|gaussian|sharpen> <3|5|7> [threads]\n",
            argv0);
}

/*
 * Rows [0, height) of one kernel. Each iteration is one row, computed by
 * convolve_rows, so the arithmetic matches the sequential program.
 *
 * Threads only write their own rows of out. in and the kernel are read.
 * Nothing else is shared, so there is no data race and no shared temporary.
 */
static int convolve_all_omp(const unsigned char *in, unsigned char *out,
                            int width, int height, const kernel_t *kernel)
{
    int failed = 0;
    #pragma omp parallel for schedule(runtime) reduction(|:failed)
    for (int y = 0; y < height; y++)
        failed |= convolve_rows(in, out, width, height, y, y + 1, kernel) != 0;
    return failed ? -1 : 0;
}

/* Same split as convolve_all_omp, but each row is sobel_rows (gx, gy, magnitude). */
static int sobel_all_omp(const unsigned char *in, unsigned char *out,
                         int width, int height, const kernel_t *gx, const kernel_t *gy)
{
    int failed = 0;
    #pragma omp parallel for schedule(runtime) reduction(|:failed)
    for (int y = 0; y < height; y++)
        failed |= sobel_rows(in, out, width, height, y, y + 1, gx, gy) != 0;
    return failed ? -1 : 0;
}

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
        usage(argv[0]);
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
        fprintf(stderr, "kernel size must be 3, 5, or 7 (got '%s')\n", argv[4]);
        usage(argv[0]);
        return 1;
    }
    int ksize = (int)ksize_long;

    int is_sobel = strcmp(filter, "sobel") == 0;
    int is_gaussian = strcmp(filter, "gaussian") == 0;
    int is_sharpen = strcmp(filter, "sharpen") == 0;
    if (!is_sobel && !is_gaussian && !is_sharpen) {
        fprintf(stderr, "unknown filter '%s'\n", filter);
        usage(argv[0]);
        return 1;
    }

    /*
     * Ask for exactly this many threads. omp_set_dynamic(0) stops the runtime
     * from quietly using fewer. The optional argument wins over OMP_NUM_THREADS.
     */
    omp_set_dynamic(0);
    if (argc == 6) {
        errno = 0;
        end = NULL;
        long n = strtol(argv[5], &end, 10);
        if (errno != 0 || end == argv[5] || *end != '\0' || n < 1 || n > INT_MAX) {
            fprintf(stderr, "thread count must be a positive integer (got '%s')\n", argv[5]);
            usage(argv[0]);
            return 1;
        }
        omp_set_num_threads((int)n);
    }
    install_schedule();
    int threads = omp_get_max_threads();

    kernel_t k_a, k_b;
    if (is_sobel) {
        if (kernel_make(K_SOBEL_X, ksize, &k_a) != 0 || kernel_make(K_SOBEL_Y, ksize, &k_b) != 0) {
            fprintf(stderr, "failed to build Sobel kernels of size %d\n", ksize);
            return 1;
        }
    } else {
        kernel_type_t type = is_gaussian ? K_GAUSSIAN : K_SHARPEN;
        if (kernel_make(type, ksize, &k_a) != 0) {
            fprintf(stderr, "failed to build %s kernel of size %d\n", filter, ksize);
            return 1;
        }
    }

    /* Total time covers I/O. Compute time starts after the image is in memory. */
    double t_total0 = omp_get_wtime();

    int width = 0, height = 0;
    long data_offset = 0;
    if (pgm_read_header(in_path, &width, &height, &data_offset) != 0) return 1;

    size_t npixels = (size_t)width * (size_t)height;
    /* Same allocation as seq_convolve: plain malloc, no pre-touch, no calloc. */
    unsigned char *in = malloc(npixels);
    unsigned char *out = malloc(npixels);
    if (!in || !out) {
        fprintf(stderr, "out of memory for a %d x %d image\n", width, height);
        free(in);
        free(out);
        return 1;
    }
    if (pgm_read_rows(in_path, data_offset, width, 0, height, in) != 0) {
        free(in);
        free(out);
        return 1;
    }

    double t_compute0 = omp_get_wtime();
    int rc;
    if (is_sobel)
        rc = sobel_all_omp(in, out, width, height, &k_a, &k_b);
    else
        rc = convolve_all_omp(in, out, width, height, &k_a);
    double t_compute1 = omp_get_wtime();
    if (rc != 0) {
        free(in);
        free(out);
        return 1;
    }

    FILE *fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "cannot open '%s' for writing: %s\n", out_path, strerror(errno));
        free(in);
        free(out);
        return 1;
    }
    long out_offset = 0;
    if (pgm_write_header(fp, width, height, &out_offset) != 0 ||
        pgm_write_rows(fp, out_offset, width, 0, height, out) != 0) {
        fclose(fp);
        free(in);
        free(out);
        return 1;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "failed to close '%s': %s\n", out_path, strerror(errno));
        free(in);
        free(out);
        return 1;
    }
    double t_total1 = omp_get_wtime();

    free(in);
    free(out);

    omp_sched_t kind = omp_sched_auto;
    int chunk = 0;
    omp_get_schedule(&kind, &chunk);

    printf("RESULT filter=%s k=%d w=%d h=%d threads=%d schedule=%s compute_s=%.3f total_s=%.3f\n",
           filter, ksize, width, height, threads, schedule_name(kind),
           t_compute1 - t_compute0, t_total1 - t_total0);
    return 0;
}

/*
 * seq_convolve.c - single-threaded convolution baseline.
 *
 * Usage:
 *   ./build/seq_convolve <in.pgm> <out.pgm> <sobel|gaussian|sharpen> <3|5|7>
 *
 * The per-pixel work lives in conv_core.h. This file only reads the image,
 * calls that code on every row, writes the image, and prints one timing line.
 * It does not use OpenMP, MPI, SIMD, or tiling. One process, one thread.
 *
 * stdout is exactly one line, for the benchmark scripts:
 *   RESULT filter=sobel k=3 w=20000 h=20000 compute_s=12.345 total_s=13.210
 * Everything else goes to stderr. compute_s is the convolution loops only;
 * total_s also includes reading the input and writing the output.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "conv_core.h"
#include "pgm_io.h"

static double monotonic_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        fprintf(stderr, "clock_gettime failed: %s\n", strerror(errno));
        exit(1);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void usage(const char *argv0)
{
    fprintf(stderr, "usage: %s <in.pgm> <out.pgm> <sobel|gaussian|sharpen> <3|5|7>\n", argv0);
}

int main(int argc, char **argv)
{
    if (argc != 5) {
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
    double t_total0 = monotonic_seconds();

    int width = 0, height = 0;
    long data_offset = 0;
    if (pgm_read_header(in_path, &width, &height, &data_offset) != 0) return 1;

    size_t npixels = (size_t)width * (size_t)height;
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

    double t_compute0 = monotonic_seconds();
    int rc;
    if (is_sobel)
        rc = sobel_rows(in, out, width, height, 0, height, &k_a, &k_b);
    else
        rc = convolve_rows(in, out, width, height, 0, height, &k_a);
    double t_compute1 = monotonic_seconds();
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
    double t_total1 = monotonic_seconds();

    free(in);
    free(out);

    printf("RESULT filter=%s k=%d w=%d h=%d compute_s=%.3f total_s=%.3f\n",
           filter, ksize, width, height,
           t_compute1 - t_compute0, t_total1 - t_total0);
    return 0;
}

/*
 * seq_convolve.c - single-threaded convolution baseline.
 *
 * Usage:
 *   ./build/seq_convolve <in.pgm> <out.pgm> <sobel|gaussian|sharpen> <3|5|7>
 *
 * Semantics are fixed by docs/DECISIONS.md:
 *   correlation (the kernel is not flipped),
 *   clamp-to-edge borders,
 *   float32 arithmetic,
 *   output byte = clamp(floor(x + 0.5), 0, 255),
 *   Sobel writes the magnitude sqrt(gx^2 + gy^2).
 *
 * This file is the honest reference for later phases. It does not use
 * OpenMP, MPI, SIMD, or tiling. One process, one thread, one pixel at a time.
 *
 * stdout is exactly one line, for the benchmark scripts:
 *   RESULT filter=sobel k=3 w=20000 h=20000 compute_s=12.345 total_s=13.210
 * Everything else goes to stderr. compute_s is the convolution loops only;
 * total_s also includes reading the input and writing the output.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernels.h"
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

/* Fold an out-of-range index onto the nearest in-range pixel (clamp-to-edge). */
static int clamp_index(int i, int limit)
{
    if (i < 0) return 0;
    if (i >= limit) return limit - 1;
    return i;
}

/*
 * One correlation at pixel (y, x).
 *
 * in points at row 0 of the FULL image. height and width are the global
 * size, so a tap that falls off the top of the picture uses row 0 even
 * when the caller is only computing a middle band of rows. That is what
 * Phase 3 needs: the halo at the global border is a copy of the edge row,
 * not a copy of the edge of this row range.
 */
static float correlate_pixel(const unsigned char *in, int width, int height,
                             int y, int x, const kernel_t *k)
{
    int radius = k->radius;
    int size = k->size;
    float acc = 0.0f;
    for (int dy = 0; dy < size; dy++) {
        int yy = clamp_index(y + dy - radius, height);
        for (int dx = 0; dx < size; dx++) {
            int xx = clamp_index(x + dx - radius, width);
            float sample = (float)in[(size_t)yy * (size_t)width + (size_t)xx];
            acc += k->w[dy * size + dx] * sample;
        }
    }
    return acc;
}

/* docs/DECISIONS.md: clamp(floor(x + 0.5), 0, 255), computed in float32. */
static unsigned char round_u8(float x)
{
    float y = floorf(x + 0.5f);
    if (y < 0.0f) y = 0.0f;
    if (y > 255.0f) y = 255.0f;
    return (unsigned char)y;
}

static int range_ok(int width, int height, int row_start, int row_end)
{
    if (width <= 0 || height <= 0 || row_start < 0 || row_end > height || row_start > row_end)
        return 0;
    return 1;
}

/*
 * Apply one kernel to rows [row_start, row_end) of a full image.
 *
 * in  and out both hold height * width bytes, row-major.
 * Only out rows in the range are written. Reads may touch any input row,
 * because the kernel hangs off the range and borders clamp to the global edge.
 * Returns 0 on success, -1 on a bad argument.
 */
int convolve_rows(const unsigned char *in, unsigned char *out,
                  int width, int height, int row_start, int row_end,
                  const kernel_t *kernel)
{
    if (!in || !out || !kernel || !range_ok(width, height, row_start, row_end)) {
        fprintf(stderr, "convolve_rows: invalid arguments\n");
        return -1;
    }
    for (int y = row_start; y < row_end; y++) {
        for (int x = 0; x < width; x++) {
            float acc = correlate_pixel(in, width, height, y, x, kernel);
            out[(size_t)y * (size_t)width + (size_t)x] = round_u8(acc);
        }
    }
    return 0;
}

/*
 * Sobel on the same row range. Magnitude needs gx and gy at the same pixel,
 * so this is convolve_rows with two kernels and a sqrt instead of one write.
 * gx and gy must be the same size (3, 5, or 7).
 */
int sobel_rows(const unsigned char *in, unsigned char *out,
               int width, int height, int row_start, int row_end,
               const kernel_t *gx, const kernel_t *gy)
{
    if (!in || !out || !gx || !gy || !range_ok(width, height, row_start, row_end)) {
        fprintf(stderr, "sobel_rows: invalid arguments\n");
        return -1;
    }
    if (gx->size != gy->size) {
        fprintf(stderr, "sobel_rows: gx and gy sizes differ\n");
        return -1;
    }
    for (int y = row_start; y < row_end; y++) {
        for (int x = 0; x < width; x++) {
            float sx = correlate_pixel(in, width, height, y, x, gx);
            float sy = correlate_pixel(in, width, height, y, x, gy);
            float mag = sqrtf(sx * sx + sy * sy);
            out[(size_t)y * (size_t)width + (size_t)x] = round_u8(mag);
        }
    }
    return 0;
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

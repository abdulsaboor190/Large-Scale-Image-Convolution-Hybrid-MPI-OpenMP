/*
 * conv_core.h - the convolution arithmetic, shared by every phase.
 *
 * Moved here unchanged from seq_convolve.c so the OpenMP program can call
 * the same functions. The math is the contract in docs/DECISIONS.md:
 *   correlation (the kernel is not flipped),
 *   clamp-to-edge borders,
 *   float32 arithmetic,
 *   output byte = clamp(floor(x + 0.5), 0, 255),
 *   Sobel magnitude sqrt(gx^2 + gy^2).
 *
 * These functions are static because this is a header included by more
 * than one program. Each program gets its own copy; the operations are
 * the same. Do not change the loops: a later phase must stay bit-identical
 * to build/seq_convolve.
 */
#ifndef CONV_CORE_H
#define CONV_CORE_H

#include <math.h>
#include <stdio.h>

#include "kernels.h"

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
static int convolve_rows(const unsigned char *in, unsigned char *out,
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
static int sobel_rows(const unsigned char *in, unsigned char *out,
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

#endif /* CONV_CORE_H */

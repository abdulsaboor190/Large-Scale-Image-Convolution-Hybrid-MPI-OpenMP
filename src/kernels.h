/*
 * kernels.h - convolution kernel definitions shared by every phase.
 *
 * Supported: sobel_x, sobel_y, gaussian, sharpen, each in 3x3, 5x5, 7x7.
 *   sobel    : outer product of a binomial smoothing row and a derivative row
 *              (standard extended Sobel). sobel_y is the transpose of sobel_x.
 *   gaussian : normalised binomial approximation (rows of Pascal's triangle).
 *   sharpen  : classic [0 -1 0; -1 5 -1; 0 -1 0] for 3x3,
 *              unsharp mask (2*identity - gaussian) for 5x5 and 7x7.
 *
 * Convention: weights are applied as CORRELATION (kernel is NOT flipped).
 * The Python reference in scripts/refconv.py uses the same definitions.
 */
#ifndef KERNELS_H
#define KERNELS_H

#include <string.h>

#define MAX_KSIZE 7

typedef enum { K_SOBEL_X, K_SOBEL_Y, K_GAUSSIAN, K_SHARPEN } kernel_type_t;

typedef struct {
    const char *name;
    int size;    /* odd: 3, 5 or 7 */
    int radius;  /* size / 2 = number of halo rows needed per side */
    float w[MAX_KSIZE * MAX_KSIZE]; /* row-major, size*size used */
} kernel_t;

static inline void kernels_binomial(int n, float *out)
{
    float row[MAX_KSIZE] = {1};
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0; j--)
            row[j] += row[j - 1];
    memcpy(out, row, n * sizeof(float));
}

/* Returns 0 on success, -1 on unsupported size. */
static inline int kernel_make(kernel_type_t type, int size, kernel_t *k)
{
    static const char *names[] = {"sobel_x", "sobel_y", "gaussian", "sharpen"};
    static const float deriv3[3] = {-1, 0, 1};
    static const float deriv5[5] = {-1, -2, 0, 2, 1};
    static const float deriv7[7] = {-1, -4, -5, 0, 5, 4, 1};

    if (size != 3 && size != 5 && size != 7) return -1;
    memset(k, 0, sizeof *k);
    k->name = names[type];
    k->size = size;
    k->radius = size / 2;

    float b[MAX_KSIZE], norm = 0.0f;
    kernels_binomial(size, b);
    for (int i = 0; i < size; i++) norm += b[i];
    norm *= norm; /* sum of the 2-D binomial kernel */

    const float *d = size == 3 ? deriv3 : size == 5 ? deriv5 : deriv7;

    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            float v = 0.0f;
            switch (type) {
            case K_SOBEL_X: v = b[r] * d[c]; break;
            case K_SOBEL_Y: v = d[r] * b[c]; break;
            case K_GAUSSIAN: v = b[r] * b[c] / norm; break;
            case K_SHARPEN:
                if (size == 3) {
                    int centre = (r == 1 && c == 1);
                    int cross = (r == 1) != (c == 1);
                    v = centre ? 5.0f : cross ? -1.0f : 0.0f;
                } else {
                    v = -b[r] * b[c] / norm;
                    if (r == size / 2 && c == size / 2) v += 2.0f;
                }
                break;
            }
            k->w[r * size + c] = v;
        }
    }
    return 0;
}

#endif /* KERNELS_H */

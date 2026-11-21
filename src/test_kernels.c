/*
 * test_kernels.c - prints every kernel as text and sanity-checks its sums.
 * scripts/verify_kernels.py parses this output and compares it to the Python
 * reference, so C and Python are guaranteed to use the same kernels.
 *
 * Sums: sobel = 0, gaussian = 1, sharpen = 1.
 */
#include <math.h>
#include <stdio.h>
#include "kernels.h"

int main(void)
{
    int bad = 0;
    for (int t = K_SOBEL_X; t <= K_SHARPEN; t++) {
        for (int size = 3; size <= 7; size += 2) {
            kernel_t k;
            if (kernel_make((kernel_type_t)t, size, &k) != 0) return 2;
            double sum = 0;
            printf("%s %d", k.name, k.size);
            for (int i = 0; i < size * size; i++) {
                printf(" %.9g", k.w[i]);
                sum += k.w[i];
            }
            printf("\n");
            double expect = (t == K_SOBEL_X || t == K_SOBEL_Y) ? 0.0 : 1.0;
            if (fabs(sum - expect) > 1e-5) {
                fprintf(stderr, "BAD SUM %s %dx%d: %f (expected %f)\n", k.name, size, size, sum, expect);
                bad = 1;
            }
        }
    }
    return bad;
}

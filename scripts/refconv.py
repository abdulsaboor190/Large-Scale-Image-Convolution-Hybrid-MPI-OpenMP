"""
Reference convolution in NumPy. This is the ground truth for the C code.

Definitions (must match src/kernels.h and docs/DECISIONS.md):
  * correlation (kernel not flipped)
  * border policy: clamp-to-edge (replicate the border pixel)
  * accumulation in float64
  * output = clamp(floor(x + 0.5), 0, 255)
  * sobel output = gradient magnitude sqrt(gx^2 + gy^2), then the same clamp
"""
import numpy as np

MAX_K = 7


def binomial(n):
    row = [1]
    for _ in range(n - 1):
        row = [a + b for a, b in zip([0] + row, row + [0])]
    return np.array(row, dtype=np.float64)


_DERIV = {
    3: np.array([-1, 0, 1], dtype=np.float64),
    5: np.array([-1, -2, 0, 2, 1], dtype=np.float64),
    7: np.array([-1, -4, -5, 0, 5, 4, 1], dtype=np.float64),
}


def make_kernel(name, size):
    b = binomial(size)
    norm = b.sum() ** 2
    if name == "sobel_x":
        return np.outer(b, _DERIV[size])
    if name == "sobel_y":
        return np.outer(_DERIV[size], b)
    if name == "gaussian":
        return np.outer(b, b) / norm
    if name == "sharpen":
        if size == 3:
            return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
        k = -np.outer(b, b) / norm
        k[size // 2, size // 2] += 2.0
        return k
    raise ValueError(name)


def correlate_clamp(img, k):
    """Same-size correlation with clamp-to-edge borders, float64 result."""
    r = k.shape[0] // 2
    h, w = img.shape
    p = np.pad(img.astype(np.float64), r, mode="edge")
    out = np.zeros((h, w), dtype=np.float64)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            out += k[dy, dx] * p[dy:dy + h, dx:dx + w]
    return out


def to_u8(x):
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)


def apply(img, filt, size):
    """filt in {'sobel', 'gaussian', 'sharpen'}; returns uint8 image."""
    if filt == "sobel":
        gx = correlate_clamp(img, make_kernel("sobel_x", size))
        gy = correlate_clamp(img, make_kernel("sobel_y", size))
        return to_u8(np.hypot(gx, gy))
    return to_u8(correlate_clamp(img, make_kernel(filt, size)))

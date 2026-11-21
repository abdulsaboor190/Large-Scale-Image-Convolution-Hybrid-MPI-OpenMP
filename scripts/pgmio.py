"""Minimal binary PGM (P5, 8-bit grayscale) reader/writer. No dependencies but numpy."""
import numpy as np


def read_pgm(path):
    with open(path, "rb") as f:
        tokens = []
        while len(tokens) < 4:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            line = line.split(b"#", 1)[0]
            tokens += line.split()
        magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
        if magic != b"P5" or maxval != 255:
            raise ValueError(f"{path}: only 8-bit binary PGM (P5, maxval 255) is supported")
        data = np.frombuffer(f.read(w * h), dtype=np.uint8)
        if data.size != w * h:
            raise ValueError(f"{path}: expected {w*h} pixels, got {data.size}")
    return data.reshape(h, w)


def pgm_header(w, h):
    return f"P5\n{w} {h}\n255\n".encode()


def write_pgm(path, img):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    with open(path, "wb") as f:
        f.write(pgm_header(w, h))
        f.write(img.tobytes())

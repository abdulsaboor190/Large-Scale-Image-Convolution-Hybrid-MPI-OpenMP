/*
 * pgm_io.h - header-only reader and writer for 8-bit binary PGM (P5).
 *
 * This is the only image format the project uses (see docs/DECISIONS.md).
 * There is no global state: every call takes a path or a FILE the caller owns.
 * Phase 3 reuses the row-range functions so one MPI rank can read or write
 * just its own rows. Row ranges are half-open: [row_start, row_end).
 *
 * Header layout we accept (same rules as scripts/pgmio.py):
 *   magic "P5", width, height, maxval 255, then raw pixels (one byte each,
 *   row-major, top to bottom). A '#' starts a comment that runs to the end
 *   of the line. Comments may sit on their own line or after a header value.
 *   Pixel bytes begin on the line after the one that contains maxval.
 */
#ifndef PGM_IO_H
#define PGM_IO_H

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Pull the four header tokens the same way scripts/pgmio.py does:
 * read a line, drop from '#' to the end of the line, then split on whitespace.
 * After the line that supplies maxval, the file position is the first pixel.
 * Header lines are short; a line that does not fit in the buffer is rejected
 * so we never accidentally swallow pixel bytes.
 */
static int pgm_read_header_tokens(FILE *fp, char tokens[4][64])
{
    int n = 0;
    char line[2048];
    while (n < 4) {
        if (!fgets(line, sizeof line, fp)) return -1;
        size_t len = strlen(line);
        if (len == sizeof line - 1 && line[len - 1] != '\n') return -1;
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *p = line;
        while (*p && n < 4) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;
            size_t i = 0;
            while (*p && !isspace((unsigned char)*p)) {
                if (i + 1 >= 64) return -1;
                tokens[n][i++] = *p++;
            }
            tokens[n][i] = '\0';
            n++;
        }
    }
    return 0;
}

static int pgm_parse_int(const char *tok, int *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(tok, &end, 10);
    if (errno != 0 || end == tok || *end != '\0' || v <= 0 || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

/*
 * Read width, height, and the byte offset of the first pixel.
 * The file is closed before return. maxval must be 255.
 * Returns 0 on success, -1 on failure (message on stderr).
 */
static int pgm_read_header(const char *path, int *width, int *height, long *data_offset)
{
    if (!path || !width || !height || !data_offset) {
        fprintf(stderr, "pgm_io: pgm_read_header missing argument\n");
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "pgm_io: cannot open '%s' for reading: %s\n", path, strerror(errno));
        return -1;
    }

    char tok[4][64];
    if (pgm_read_header_tokens(fp, tok) != 0) {
        fprintf(stderr, "pgm_io: '%s' has a truncated or over-long header\n", path);
        fclose(fp);
        return -1;
    }
    if (strcmp(tok[0], "P5") != 0) {
        fprintf(stderr, "pgm_io: '%s' is not a binary PGM (expected magic P5)\n", path);
        fclose(fp);
        return -1;
    }
    if (pgm_parse_int(tok[1], width) != 0) {
        fprintf(stderr, "pgm_io: '%s' has a bad width\n", path);
        fclose(fp);
        return -1;
    }
    if (pgm_parse_int(tok[2], height) != 0) {
        fprintf(stderr, "pgm_io: '%s' has a bad height\n", path);
        fclose(fp);
        return -1;
    }

    int maxval = 0;
    if (pgm_parse_int(tok[3], &maxval) != 0 || maxval != 255) {
        fprintf(stderr, "pgm_io: '%s' must be 8-bit (maxval 255)\n", path);
        fclose(fp);
        return -1;
    }

    long off = ftell(fp);
    if (off < 0) {
        fprintf(stderr, "pgm_io: '%s': ftell failed: %s\n", path, strerror(errno));
        fclose(fp);
        return -1;
    }
    if ((size_t)(*height) > SIZE_MAX / (size_t)(*width)) {
        fprintf(stderr, "pgm_io: '%s' is too large to address (%d x %d)\n", path, *width, *height);
        fclose(fp);
        return -1;
    }

    *data_offset = off;
    if (fclose(fp) != 0) {
        fprintf(stderr, "pgm_io: '%s': fclose failed: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

/*
 * Read rows [row_start, row_end) into dst.
 * dst must hold (row_end - row_start) * width bytes.
 * data_offset and width come from pgm_read_header.
 * Returns 0 on success, -1 on failure (message on stderr).
 */
static int pgm_read_rows(const char *path, long data_offset, int width,
                         int row_start, int row_end, unsigned char *dst)
{
    if (!path || !dst || data_offset < 0 || width <= 0 || row_start < 0 || row_end < row_start) {
        fprintf(stderr, "pgm_io: pgm_read_rows bad argument\n");
        return -1;
    }
    if (row_start == row_end) return 0;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "pgm_io: cannot open '%s' for reading: %s\n", path, strerror(errno));
        return -1;
    }

    long pos = data_offset + (long)row_start * (long)width;
    if (fseek(fp, pos, SEEK_SET) != 0) {
        fprintf(stderr, "pgm_io: '%s': seek to row %d failed: %s\n", path, row_start, strerror(errno));
        fclose(fp);
        return -1;
    }

    size_t nbytes = (size_t)(row_end - row_start) * (size_t)width;
    size_t got = fread(dst, 1, nbytes, fp);
    if (got != nbytes) {
        fprintf(stderr, "pgm_io: '%s': short read at rows [%d, %d) (%zu of %zu bytes)\n",
                path, row_start, row_end, got, nbytes);
        fclose(fp);
        return -1;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "pgm_io: '%s': fclose failed: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

/*
 * Write a P5 header to an already-open binary FILE.
 * On success, *data_offset is the byte position where row 0 starts.
 * Returns 0 on success, -1 on failure (message on stderr).
 */
static int pgm_write_header(FILE *fp, int width, int height, long *data_offset)
{
    if (!fp || !data_offset || width <= 0 || height <= 0) {
        fprintf(stderr, "pgm_io: pgm_write_header bad argument\n");
        return -1;
    }
    if (fprintf(fp, "P5\n%d %d\n255\n", width, height) < 0) {
        fprintf(stderr, "pgm_io: failed to write header: %s\n", strerror(errno));
        return -1;
    }
    long off = ftell(fp);
    if (off < 0) {
        fprintf(stderr, "pgm_io: ftell after header failed: %s\n", strerror(errno));
        return -1;
    }
    *data_offset = off;
    return 0;
}

/*
 * Write rows [row_start, row_end) at the matching place in an open PGM.
 * src holds (row_end - row_start) * width bytes (the rows themselves, not a
 * full image). data_offset is the value pgm_write_header stored.
 * Seeking makes it safe to write ranges out of order.
 * Returns 0 on success, -1 on failure (message on stderr).
 */
static int pgm_write_rows(FILE *fp, long data_offset, int width,
                          int row_start, int row_end, const unsigned char *src)
{
    if (!fp || !src || data_offset < 0 || width <= 0 || row_start < 0 || row_end < row_start) {
        fprintf(stderr, "pgm_io: pgm_write_rows bad argument\n");
        return -1;
    }
    if (row_start == row_end) return 0;

    long pos = data_offset + (long)row_start * (long)width;
    if (fseek(fp, pos, SEEK_SET) != 0) {
        fprintf(stderr, "pgm_io: seek to output row %d failed: %s\n", row_start, strerror(errno));
        return -1;
    }

    size_t nbytes = (size_t)(row_end - row_start) * (size_t)width;
    size_t put = fwrite(src, 1, nbytes, fp);
    if (put != nbytes) {
        fprintf(stderr, "pgm_io: short write of rows [%d, %d) (%zu of %zu bytes): %s\n",
                row_start, row_end, put, nbytes, strerror(errno));
        return -1;
    }
    return 0;
}

#endif /* PGM_IO_H */

#include <stdio.h>
#include <string.h>

#include "bov.h"

static void header0(FILE *f, const char *val,
                    int ox, int oy, int oz,
                    int sx, int sy, int sz,
                    int ncomp) {
    fprintf(f, "DATA_FILE: %s\n", val);
    fprintf(f, "DATA_SIZE: %d %d %d\n", sx, sy, sz);
    fprintf(f, "VARIABLE: D\n");
    fprintf(f, "CENTERING: zonal\n");
    fprintf(f, "BRICK_ORIGIN: %d %d %d\n", ox, oy, oz);
    fprintf(f, "BRICK_SIZE:   %d %d %d\n", sx, sy, sz);
    fprintf(f, "DATA_COMPONENTS: %d\n", ncomp);
}

static void header(const char *bov, const char *val,
                   int ox, int oy, int oz,
                   int sx, int sy, int sz, int ncomp) {
    FILE *f  = fopen(bov, "w");
    header0(f, val, ox, oy, oz, sx, sy, sz, ncomp);
    fclose(f);
}

static void data(const char *b, float *D, size_t size) {
    FILE *f = fopen(b, "w");
    fwrite(D, sizeof(float), size, f);
    fclose(f);
}

void write(const char *b, float *D,
           int ox, int oy, int oz,
           int sx, int sy, int sz,
           int ncomp) {
    char bov[BUFSIZ], val[BUFSIZ];
    size_t size;

    strcpy(bov, b); strcat(bov, ".bov");
    strcpy(val, b); strcat(val, ".value");

    header(bov, val, ox, oy, oz, sx, sy, sz, ncomp);
    size = sx * sy * sz * ncomp;
    data(val, D, size);
}

#include <stdio.h>
#include <string.h>

#define NX 5
#define NY 5
#define NZ 5
#define NCOMP 1

void header0(FILE *f, const char *bov,
             int ox, int oy, int oz,
             int sx, int sy, int sz,
             int ncomp) {
    fprintf(f, "DATA_FILE: bov.values\n");
    fprintf(f, "DATA_SIZE: %d %d %d\n", sx, sy, sz);
    fprintf(f, "VARIABLE: D\n");
    fprintf(f, "CENTERING: zonal\n");
    fprintf(f, "BRICK_ORIGIN: %d %d %d\n", ox, oy, oz);
    fprintf(f, "BRICK_SIZE:   %d %d %d\n", sx, sy, sz);
    fprintf(f, "DATA_COMPONENTS: %d\n", ncomp);
}

void header(const char *bov, const char *val,
            int ox, int oy, int oz,
            int sx, int sy, int sz, int ncomp) {
    FILE *f  = fopen(bov, "w");
    header0(f, val, ox, oy, oz, sx, sy, sz, ncomp);
    fclose(f);
}

void data(const char *b, float *D, int size) {
    FILE *f = fopen(b, "w");
    fwrite(D, sizeof(float), size, f);
    fclose(f);
}

void write(const char *b, float *D,
           int ox, int oy, int oz,
           int sx, int sy, int sz,
           int ncomp) {
    char bov[BUFSIZ], val[BUFSIZ];
    int size;
    
    strcpy(bov, b); strcat(bov, ".bov");
    strcpy(val, b); strcat(val, ".value");

    header(bov, val, ox, oy, oz, sx, sy, sz, ncomp);
    size = sx * sy * sz * ncomp;
    data(val, D, size);
}

int main() {
    /* float data[NZ][NY][NX][NCOMP]; */
    float D[NX*NY*NZ*NCOMP];
    write("main", D,
          0,   0,  0,
          NX, NY, NZ,
          NCOMP);
}

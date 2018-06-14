#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "type.h"
#include "com.h"
#include "disp.h"

enum {X, Y, Z, D};

struct Arg {
    int L[D];
    int nfiles;
    char **fnames;
    char *op;
};

static void usg() {
    fprintf(stderr,
            "usage: u.post.rbc.flux <Lx> <Ly> <Lz> <op> <com-00.txt> <com-01.txt> ...\n"
            "\t Lx, Ly, Lz: periodic domain size\n"
            "\t op: -sum, -ratio, -maxy \n");
    
    exit(1);
}

static bool shift(int *c, char ***v) {
    (*c) --;
    (*v) ++;
    return *c > 0;
}

static void parse(int c, char **v, Arg *a) {
    if (!shift(&c, &v)) usg();
    a->L[X] = atof(*v);
    if (!shift(&c, &v)) usg();
    a->L[Y] = atof(*v);
    if (!shift(&c, &v)) usg();
    a->L[Z] = atof(*v);

    if (!shift(&c, &v)) usg();
    a->op = *v;
    
    if (!shift(&c, &v)) usg();
    a->fnames = v;
    a->nfiles = c;

    if (c < 2) usg();        
}

template <typename T>
static void swap(T *a, T *b) {
    T c = *a;
    *a = *b;
    *b = c;
}

static void op_sum(float dx, float dy, float dz, float *res) {
    res[X] += dx;
    res[Y] += dy;
    res[Z] += dz;
}

static void op_sum_ratio(float dx, float dy, float dz, float *res) {
    *res += dy / dx;
}

static void op_maxy(float dx, float dy, float dz, float *res) {
    if (fabs(dy) > fabs(*res))
        *res = dy;
}

int main(int argc, char **argv) {
    Arg a;
    Disp *d;
    Com *cc0, *cc1;
    int i, n;
    FILE *f;

    parse(argc, argv, &a);

    f = fopen(a.fnames[0], "r");
    read(f, &n, &cc0);
    sort_by_id(n, cc0);
    fclose(f);

    disp_ini(n, &d);
    
    for (i = 1; i < a.nfiles; ++i) {
        f = fopen(a.fnames[i], "r");
        read(f, &n, &cc1);
        fclose(f);
        
        sort_by_id(n, cc1);

        disp_add(n, cc0, cc1, a.L, d);
        
        swap(&cc0, &cc1);
        
        free(cc1);
    }
    free(cc0);

    float s = 1.0 / n;

    if (0 == strcmp(a.op, "-sum")) {
        float tot[D] = {0};
        disp_reduce(d, &op_sum, tot);
        printf("%g %g %g\n", s*tot[X], s*tot[Y], s*tot[Z]);
    }    
    else if (0 == strcmp(a.op, "-maxy")) {
        float max_dy = 0;
        disp_reduce(d, &op_maxy, &max_dy);
        printf("%g\n", max_dy);
    }
    else if (0 == strcmp(a.op, "-ratio")) {
        float ratio = 0;
        disp_reduce(d, &op_sum_ratio, &ratio);
        printf("%g\n", s*ratio);
    }
    else {
        printf("unknown operation %s\n", a.op);
    }
    disp_fin(d);
    
    return 0;
}

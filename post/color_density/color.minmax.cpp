#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "bov.h"
#include "bov_serial.h"

struct Args {
    char *bov;
    float T0, T1, XT;
};

static void usg() {
    fprintf(stderr, "usg: u.color.minmax <XT> <T0> <T1> <solvent.bop>\n");
    exit(1);
}

static void parse(int argc, char **argv, /**/ Args *a) {
    if (argc != 5) usg();
    int iarg = 1;
    a->XT  = atof(argv[iarg++]);
    a->T0  = atof(argv[iarg++]);
    a->T1  = atof(argv[iarg++]);
    a->bov =      argv[iarg++];
}

static void minmax(int nx, int ny, const float *d, float T0, float T1, float XT, /**/ float *minx, float *maxx) {
    int ix, iy, i;
    float xmin, xmax;

    xmin = nx;
    xmax = 0;
    
    for (i = iy = 0; iy < ny; ++iy) {
        for (ix = 0; ix < nx; ++ix, ++i) {
            
            if (ix > XT && ix < xmin && d[i] < T0) xmin = ix;
            if (ix > XT && ix > xmax && d[i] > T1) xmax = ix;
        }
    }
    if (xmax < xmin) xmax = xmin;
    
    *minx = xmin;
    *maxx = xmax;
}

int main(int argc, char **argv) {
    Args a;
    BovDesc bov;
    const float *grid;
    char fdname[256];
    float xmin, xmax;
    
    parse(argc, argv, /**/ &a);
    
    bov_read_header(a.bov, /**/ &bov, fdname);
    bov_alloc(sizeof(float), /**/ &bov);
    bov_read_values(fdname, /**/ &bov);

    assert(bov.nz == 1);
    
    grid = (const float*) bov.data;

    minmax(bov.nx, bov.ny, grid, a.T0, a.T1, a.XT, /**/ &xmin, &xmax); 

    printf("%g %g\n", xmin, xmax);
    
    bov_free(&bov);

    return 0;
}

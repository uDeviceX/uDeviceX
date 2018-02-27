#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "bov_common.h"
#include "bov_serial.h"
#include "../common/macros.h"

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
    enum {X, Y, Z, D};
    Args a;
    BovData *bov;
    const float *grid;
    char fdname[256];
    float xmin, xmax;
    int n[D];
    
    parse(argc, argv, /**/ &a);

    BVC( bov_ini(&bov) );

    BVC( bov_read_header(a.bov, /**/ bov, fdname) );
    BVC( bov_alloc(/**/ bov) );
    BVC( bov_read_values(fdname, /**/ bov) );
    
    grid = (const float*) bov_get_data(bov);

    BVC( bov_get_gridsize(bov, n) );

    minmax(n[X], n[Y], grid, a.T0, a.T1, a.XT, /**/ &xmin, &xmax); 

    printf("%g %g\n", xmin, xmax);
    
    BVC( bov_fin(bov) );

    return 0;
}

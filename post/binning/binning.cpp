#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "bop_common.h"
#include "bop_serial.h"

#include "bov.h"
#include "bov_serial.h"

struct Args {
    float lx, ly, lz;
    int nx, ny, nz;
    char *bop, *bov;
    char *field;
};

static void usg() {
    fprintf(stderr, "usg: u.avg nx ny nz Lx Ly Lz <solvent.bop> <out> <u/v/w/rho>\n");
    exit(1);
}

static void parse(int argc, char **argv, /**/ Args *a) {
    if (argc != 10) usg();
    int iarg = 1;
    a->nx = atoi(argv[iarg++]);
    a->ny = atoi(argv[iarg++]);
    a->nz = atoi(argv[iarg++]);

    a->lx = atof(argv[iarg++]);
    a->ly = atof(argv[iarg++]);
    a->lz = atof(argv[iarg++]);

    a->bop = argv[iarg++];
    a->bov = argv[iarg++];

    a->field = argv[iarg++];
}

int main(int argc, char **argv) {
    Args a;
    BopData bop;
    BovDesc bov;
    float *grid, dx, dy, dz;
    char fdname[CBUFSIZE];
    size_t sz;
    
    parse(argc, argv, /**/ &a);

    sz = a.nx * a.ny * a.nz * sizeof(float);
    grid = (float*) malloc(sz);
    
    bop_read_header(a.bop, /**/ &bop, fdname);
    bop_alloc(/**/ &bop);
    bop_read_values(fdname, /**/ &bop);

    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;
    
    // avg(bop.n, (const float*) bop_s.data, (const int*) bop_c.data,
    //             a.nx, a.ny, a.nz, dx, dy, dz, /**/ grid);    
    
    bov.nx = a.nx; bov.ny = a.ny; bov.nz = a.nz;
    bov.lx = a.lx; bov.ly = a.ly; bov.lz = a.lz;
    bov.ox = bov.oy = bov.oz = 0.f;
    bov.data = grid;
    sprintf(bov.var, a.field);
    bov.ncmp = 1;

    bov_alloc(sizeof(float), &bov);

    memcpy(bov.data, grid, sz);

    bov_write_header(a.bov, &bov);
    bov_write_values(a.bov, &bov);
    
    free(grid);
    
    bop_free(&bop);
    bov_free(&bov);

    return 0;
}

/*

  # snTEST: u.t0
  # make 
  # t=grid
  # ./avg 16 32 12 16 32 12 data/test.bop data/colors.bop $t
  # bop2txt $t.bov > colden.out.txt

*/

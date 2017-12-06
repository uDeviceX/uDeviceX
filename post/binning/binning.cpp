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
    fprintf(stderr, "usg: u.binning nx ny nz Lx Ly Lz <solvent.bop> <out> <u/v/w/rho>\n");
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

enum {INVALID = -1};

static int r2cid(const float r[3],
                 int nx, int ny, int nz,
                 float dx, float dy, float dz,
                 float ox, float oy, float oz) {
    enum {X, Y, Z};
    int ix, iy, iz;
    ix = (r[X] - ox) / dx;
    iy = (r[Y] - oy) / dy;
    iz = (r[Z] - oz) / dz;

    if (ix < 0 || ix >= nx ||
        iy < 0 || iy >= ny ||
        iz < 0 || iz >= nz)
        return INVALID;

    return ix + nx * (iy + ny * iz);
}

static void binning(int n, const float *pp,
                    int nx, int ny, int nz,
                    float dx, float dy, float dz,
                    float ox, float oy, float oz,
                    /**/ float *grid, int *counts) {

    int i, cid;
    const float *r, *u;
    
    for (i = 0; i < n; ++i) {
        r = pp + 6 * i + 0;
        u = pp + 6 * i + 3;
        cid = r2cid(r, nx, ny, nz, dx, dy, dz, ox, oy, oz);

        if (cid != INVALID) {
            counts[cid] ++;
            grid[cid] += u[0]; // TODO
        }
    }
}

static void avg(int n, const int *counts, /**/ float *grid) {
    int i, c;
    float s;
    for (i = 0; i < n; ++i) {
        c = counts[i];
        s = c ? 1.f / c : 1;
        grid[i] *= s;
    }
}

int main(int argc, char **argv) {
    Args a;
    BopData bop;
    BovDesc bov;
    float *grid, dx, dy, dz;
    int ngrid, *counts;
    char fdname[CBUFSIZE];
    size_t sz;
    
    parse(argc, argv, /**/ &a);

    ngrid = a.nx * a.ny * a.nz;
    
    sz = ngrid * sizeof(float);
    grid = (float*) malloc(sz);
    memset(grid, 0, sz);
    
    sz = ngrid * sizeof(float);
    counts = (int*) malloc(sz);
    memset(counts, 0, sz);
    
    bop_read_header(a.bop, /**/ &bop, fdname);
    bop_alloc(/**/ &bop);
    bop_read_values(fdname, /**/ &bop);

    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;

    binning(bop.n, (const float*) bop.data,
            a.nx, a.ny, a.nz,
            dx, dy, dz, 0, 0, 0, /**/ grid, counts);    

    avg(ngrid, counts, /**/ grid);
    
    bov.nx = a.nx; bov.ny = a.ny; bov.nz = a.nz;
    bov.lx = a.lx; bov.ly = a.ly; bov.lz = a.lz;
    bov.ox = bov.oy = bov.oz = 0.f;
    bov.data = grid;
    sprintf(bov.var, a.field);
    bov.ncmp = 1;

    bov_alloc(sizeof(float), &bov);

    memcpy(bov.data, grid, ngrid * sizeof(float));

    bov_write_header(a.bov, &bov);
    bov_write_values(a.bov, &bov);
    
    free(grid);
    free(counts);
    
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

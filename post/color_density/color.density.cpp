#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "bop_common.h"
#include "bop_serial.h"

#include "bov_common.h"
#include "bov_serial.h"

#include "../common/macros.h"

struct Args {
    float lx, ly, lz;
    int nx, ny, nz;
    char *bop_s, *bop_c, *bov;
};

static void usg() {
    fprintf(stderr, "usg: u.color.density nx ny nz Lx Ly Lz <solvent.bop> <colors.bop> <out>\n");
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

    a->bop_s = argv[iarg++];
    a->bop_c = argv[iarg++];
    a->bov   = argv[iarg++];
}

static void p2m_1cid(float wx, float wy, float wz, int ix, int iy, int iz, int nx, int ny, /**/ float *grid) {
    int i;
    i = ix + nx * (iy + ny * iz);
    grid[i] += wx * wy * wz;
}

static void collect_p2m(long n, const float *pp, const int *cc,
                        int nx, int ny, int nz, float dx, float dy, float dz, /**/ float *grid) {
    enum {X, Y, Z};
    long i, ix0, iy0, iz0, ix1, iy1, iz1;
    float x, y, z; // weights
    const float *r;

    memset(grid, 0, nx * ny * nz * sizeof(float));

    for (i = 0; i < n; ++i) {
        if (cc[i] == 0) continue;

        r = pp + 6 * i;

        ix0 = (int) (r[X] / dx);
        iy0 = (int) (r[Y] / dy);
        iz0 = (int) (r[Z] / dz);

        ix1 = (ix0 + 1) % nx;
        iy1 = (iy0 + 1) % ny;
        iz1 = (iz0 + 1) % nz;
        
        x = (r[X] - ix0 * dx) / dx;
        y = (r[Y] - iy0 * dy) / dy;
        z = (r[Z] - iz0 * dz) / dz;

        p2m_1cid(1.f - x, 1.f - y, 1.f - z,     ix0, iy0, iz0,    nx, ny, /**/ grid);
        p2m_1cid(      x, 1.f - y, 1.f - z,     ix1, iy0, iz0,    nx, ny, /**/ grid);
        p2m_1cid(1.f - x,       y, 1.f - z,     ix0, iy1, iz0,    nx, ny, /**/ grid);
        p2m_1cid(      x,       y, 1.f - z,     ix1, iy1, iz0,    nx, ny, /**/ grid);

        p2m_1cid(1.f - x, 1.f - y,       z,     ix0, iy0, iz1,    nx, ny, /**/ grid);
        p2m_1cid(      x, 1.f - y,       z,     ix1, iy0, iz1,    nx, ny, /**/ grid);
        p2m_1cid(1.f - x,       y,       z,     ix0, iy1, iz1,    nx, ny, /**/ grid);
        p2m_1cid(      x,       y,       z,     ix1, iy1, iz1,    nx, ny, /**/ grid);
    }
}

int main(int argc, char **argv) {
    Args a;
    BopData *bop_s, *bop_c;
    BovData *bov;
    float *grid, dx, dy, dz;
    char fdname[FILENAME_MAX];
    size_t sz;
    long np;
    const float *pp;
    const int *cc;
    
    parse(argc, argv, /**/ &a);

    sz = a.nx * a.ny * a.nz * sizeof(float);
    grid = (float*) malloc(sz);

    BPC( bop_ini(&bop_s) );
    BPC( bop_ini(&bop_c) );
    BVC( bov_ini(&bov) );
    
    BPC( bop_read_header(a.bop_s, /**/ bop_s, fdname) );
    BPC( bop_alloc(/**/ bop_s) );
    BPC( bop_read_values(fdname, /**/ bop_s) );

    BPC( bop_read_header(a.bop_c, /**/ bop_c, fdname) );
    BPC( bop_alloc(/**/ bop_c) );
    BPC( bop_read_values(fdname, /**/ bop_c) );

    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;

    BPC( bop_get_n(bop_s, &np) );
    pp = (const float*) bop_get_data(bop_s);
    cc = (const   int*) bop_get_data(bop_c);
    
    collect_p2m(np, pp, cc, a.nx, a.ny, a.nz, dx, dy, dz, /**/ grid);    

    BVC( bov_set_gridsize(a.nx, a.ny, a.nz, bov) );
    BVC( bov_set_origin(-0.5, -0.5, -0.5, bov) );
    BVC( bov_set_extent(a.lx, a.ly, a.lz, bov) );
    BVC( bov_set_ncomp(1, bov) );
    BVC( bov_set_type(BovFLOAT, bov) );
    BVC( bov_set_var("color density", bov) );

    BVC( bov_alloc(bov) );

    memcpy(bov_get_data(bov), grid, sz);
    
    BVC( bov_write_header(a.bov, bov) );
    BVC( bov_write_values(a.bov, bov) );
    
    free(grid);
    
    BPC( bop_fin(bop_s) );
    BPC( bop_fin(bop_c) );

    BVC( bov_fin(bov) );

    return 0;
}

/*

  # nTEST: colden.t0
  # make 
  # t=grid
  # ./color.density 16 32 12 16 32 12 data/test.bop data/colors.bop $t
  # bop2txt $t.bov > colden.out.txt

  # nTEST: colden.t1
  # make 
  # t=grid
  # ./color.density 32 64 24 16 32 12 data/test.bop data/colors.bop $t
  # bop2txt $t.bov > colden.out.txt

*/

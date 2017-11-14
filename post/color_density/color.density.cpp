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
    BopData bop_s, bop_c;
    BovDesc bov;
    float *grid, dx, dy, dz;
    char fdname[CBUFSIZE];
    size_t sz;
    
    parse(argc, argv, /**/ &a);

    sz = a.nx * a.ny * a.nz * sizeof(float);
    grid = (float*) malloc(sz);
    
    bop_read_header(a.bop_s, /**/ &bop_s, fdname);
    bop_alloc(/**/ &bop_s);
    bop_read_values(fdname, /**/ &bop_s);

    bop_read_header(a.bop_c, /**/ &bop_c, fdname);
    bop_alloc(/**/ &bop_c);
    bop_read_values(fdname, /**/ &bop_c);

    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;
    
    collect_p2m(bop_s.n, (const float*) bop_s.data, (const int*) bop_c.data,
                a.nx, a.ny, a.nz, dx, dy, dz, /**/ grid);    
    
    bov.nx = a.nx; bov.ny = a.ny; bov.nz = a.nz;
    bov.lx = a.lx; bov.ly = a.ly; bov.lz = a.lz;
    bov.ox = -0.5;    bov.oy = -0.5;    bov.oz = -0.5;
    bov.data = grid;
    sprintf(bov.var, "color density");
    bov.ncmp = 1;    

    bov_alloc(sizeof(float), &bov);

    memcpy(bov.data, grid, sz);

    bov_write_header(a.bov, &bov);
    bov_write_values(a.bov, &bov);
    
    free(grid);
    
    bop_free(&bop_s);
    bop_free(&bop_c);

    bov_free(&bov);

    return 0;
}

/*

  # nTEST: colden.t0
  # make 
  # t=grid
  # ./color.density 16 32 12 16 32 12 data/test.bop data/colors.bop $t
  # bop2txt $t.bov > colden.out.txt

*/

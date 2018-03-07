#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include <assert.h>

#include "bop_common.h"
#include "bop_serial.h"

#include "bov_common.h"
#include "bov_serial.h"

#include "../common/macros.h"
      
struct Args {
    float lx, ly, lz;
    int nx, ny, nz;
    char **ppp, **sss, *bov;
    int nsteps;
};

static void usg() {
    fprintf(stderr, "usg: u.stress.avg nx ny nz Lx Ly Lz <out> <pp0.bop> <pp2.bop> ... -- <ss1.bop> <ss2.bop> ...\n");
    exit(1);
}

static int shift_args(int *c, char ***v) {
    (*c) --;
    (*v) ++;
    return (*c) > 0;
}

static void parse(int argc, char **argv, /**/ Args *a) {
    int nsteps;
    
    // skip exe
    if (!shift_args(&argc, &argv)) usg();
    a->nx = atoi(*argv);

    if (!shift_args(&argc, &argv)) usg();
    a->ny = atoi(*argv);

    if (!shift_args(&argc, &argv)) usg();
    a->nz = atoi(*argv);

    if (!shift_args(&argc, &argv)) usg();
    a->lx = atof(*argv);

    if (!shift_args(&argc, &argv)) usg();
    a->ly = atof(*argv);

    if (!shift_args(&argc, &argv)) usg();
    a->lz = atof(*argv);

    if (!shift_args(&argc, &argv)) usg();
    a->bov = *argv;

    
    if (!shift_args(&argc, &argv)) usg();
    a->ppp = argv;

    nsteps = 0;
    while (strcmp("--", *argv)) {
        if (!shift_args(&argc, &argv)) usg();
        ++nsteps;
    }
    
    if (!nsteps)    usg();
    
    if (!shift_args(&argc, &argv)) usg();
    a->sss = argv;

    if (argc != nsteps) usg();
    
    a->nsteps = nsteps;
}

enum {INVALID = -1};

// bin index from position; INVALID: ignored
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

enum {
    NPP = 6, // number of components per particle
    NPS = 6  // number of components per stress
};

enum {
    XX,  XY,  XZ,  YY,  YZ,  ZZ,
    KXX, KXY, KXZ, KYY, KYZ, KZZ, NPG
};

// reduce field quantities in each cell
static void binning(long n, const float *pp, const float *ss,
                    int nx, int ny, int nz,
                    float dx, float dy, float dz,
                    float ox, float oy, float oz,
                    /**/ float *grid, int *counts) {
    enum {X, Y, Z, D};
    int i, cid;
    const float *r, *u, *s;
    float *g;
    
    for (i = 0; i < n; ++i) {
        r = pp + NPP * i;
        u = r + D;
        s = ss + NPS * i;
        
        cid = r2cid(r, nx, ny, nz, dx, dy, dz, ox, oy, oz);
        
        if (cid != INVALID) {
            g = grid + NPG * cid;
            counts[cid] ++;
            g[XX] += s[XX];
            g[XY] += s[XY];
            g[XZ] += s[XZ];
            g[YY] += s[YY];
            g[YZ] += s[YZ];
            g[ZZ] += s[ZZ];

            g[KXX] += u[X] * u[X];
            g[KXY] += u[X] * u[Y];
            g[KXZ] += u[X] * u[Z];
            g[KYY] += u[Y] * u[Y];
            g[KYZ] += u[Y] * u[Z];
            g[KZZ] += u[Z] * u[Z];
        }
    }
}

// average: divide by counts in each cell
static void avg(int n, const int *counts, float vol, /**/ float *grid) {
    int i, c, j;
    double s, svol;
    svol = 1.0 / vol;
    for (i = 0; i < n; ++i) {
        c = counts[i];
        s = c ? svol / c : svol;
        for (j = 0; j < NPG; ++j)
            grid[i * NPG + j] *= s;
    }
}

static void read_bop(const char *name, BopData *bop) {
    char fdname[FILENAME_MAX];
    BPC(bop_read_header(name, /**/ bop, fdname));
    BPC(bop_alloc(/**/ bop));
    BPC(bop_read_values(fdname, /**/ bop));
}

void add_to_grid(int i, Args a, float dx, float dy, float dz, /**/ float *grid, int *counts) {
    BopData *pp_bop, *ss_bop;
    long n, ns;
    const float *pp, *ss;
    
    BPC(bop_ini(&pp_bop));
    BPC(bop_ini(&ss_bop));

    // printf("%s -- %s\n", a.ppp[i], a.sss[i]);

    read_bop(a.ppp[i], pp_bop);
    read_bop(a.sss[i], ss_bop);
    
    BPC(bop_get_n(pp_bop, &n));
    BPC(bop_get_n(ss_bop, &ns));

    assert(n == ns);
    
    pp = (const float*) bop_get_data(pp_bop);
    ss = (const float*) bop_get_data(ss_bop);
    
    binning(n, pp, ss,
            a.nx, a.ny, a.nz,
            dx, dy, dz, 0, 0, 0,
            /**/ grid, counts);

    BPC(bop_fin(pp_bop));
    BPC(bop_fin(ss_bop));
}

void write_bov(Args a, const float *grid) {
    BovData *bov;
    long ngrid;

    ngrid = a.nx * a.ny * a.nz;
    
    BVC(bov_ini(&bov));

    BVC(bov_set_gridsize(a.nx, a.ny, a.nz, bov));
    BVC(bov_set_extent(a.lx, a.ly, a.lz, bov));
    BVC(bov_set_origin(0, 0, 0, bov));
    BVC(bov_set_var("sxx sxy sxz syy syz szz kxx kxy kxz kyy kyz kzz", bov));
    BVC(bov_set_ncomp(NPG, bov));
    BVC(bov_set_type(BovFLOAT, bov));
    
    BVC(bov_alloc(bov));

    memcpy(bov_get_data(bov), grid, ngrid * NPG * sizeof(float));

    BVC(bov_write_header(a.bov, bov));
    BVC(bov_write_values(a.bov, bov));

    BVC(bov_fin(bov));
}

int main(int argc, char **argv) {
    Args a;
    float *grid, dx, dy, dz;
    int ngrid, *counts, i;
    size_t sz;
    
    parse(argc, argv, /**/ &a);

    ngrid = a.nx * a.ny * a.nz;
    
    sz = ngrid * NPG * sizeof(float);
    grid = (float*) malloc(sz);
    memset(grid, 0, sz);
    
    sz = ngrid * sizeof(int);
    counts = (int*) malloc(sz);
    memset(counts, 0, sz);

    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;

    for (i = 0; i < a.nsteps; ++i)
        add_to_grid(i, a, dx, dy, dz, /**/ grid, counts);

    avg(ngrid, counts, dx*dy*dz, /**/ grid);

    write_bov(a, grid);
    
    free(grid);
    free(counts);
    
    return 0;
}

/*

  # TEST: avg.t0
  # rm -f *out.txt
  # make 
  # t=grid
  # ./stress.avg 3 1 1   3 3 3 $t data/pp-0.bop -- data/ss-0.bop
  # bov2txt $t.bov > ss.out.txt

  # TEST: avg.t1
  # rm -f *out.txt
  # make 
  # t=grid
  # ./stress.avg 3 1 1   3 3 3 $t data/pp-0.bop data/pp-1.bop -- data/ss-0.bop data/ss-1.bop
  # bov2txt $t.bov > ss.out.txt

*/

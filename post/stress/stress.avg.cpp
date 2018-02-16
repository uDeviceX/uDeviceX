#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

#include "bop_common.h"
#include "bop_serial.h"

#include "bov.h"
#include "bov_serial.h"

#define BPC(ans) do {                                   \
        BopStatus s = ans;                              \
        if (!bop_success(s)) {                          \
            fprintf(stderr, "%s: %d: Bop error: %s\n",  \
                    __FILE__, __LINE__,                 \
                    bob_report_error_desc(s));          \
            exit(1);                                    \
        }                                               \
    } while(0)
      
struct Args {
    float lx, ly, lz;
    int nx, ny, nz;
    char *pp, *ss, *bov;
};

static void usg() {
    fprintf(stderr, "usg: u.stress nx ny nz Lx Ly Lz <particles.bop> <stress.bop> <out>\n");
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
    
    a->pp = argv[iarg++];
    a->ss = argv[iarg++];
    a->bov = argv[iarg++];
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
static void binning(int n, const float *pp, const float *ss,
                    int nx, int ny, int nz,
                    float dx, float dy, float dz,
                    float ox, float oy, float oz,
                    /**/ float *grid, int *counts) {
    enum {X, Y, Z};
    int i, cid;
    const float *r, *u, *s;
    float *g;
    
    for (i = 0; i < n; ++i) {
        r = pp + NPP * i + 0;
        u = pp + NPP * i + 3;
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
        else {
            printf("%g %g %g\n", r[X], r[Y], r[Z]);
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

int main(int argc, char **argv) {
    Args a;
    BopData *pp_bop, *ss_bop;
    BovDesc bov;
    float *grid, dx, dy, dz;
    int ngrid, *counts;
    char fdname[Cbuf::SIZ];
    size_t sz;
    long n;
    float *pp, *ss;
    
    parse(argc, argv, /**/ &a);

    ngrid = a.nx * a.ny * a.nz;
    
    sz = ngrid * NPG * sizeof(float);
    grid = (float*) malloc(sz);
    memset(grid, 0, sz);
    
    sz = ngrid * sizeof(int);
    counts = (int*) malloc(sz);
    memset(counts, 0, sz);

    BPC(bop_ini(&pp_bop));
    BPC(bop_ini(&ss_bop));

    BPC(bop_read_header(a.pp, /**/ pp_bop, fdname));
    BPC(bop_alloc(/**/ pp_bop));
    BPC(bop_read_values(fdname, /**/ pp_bop));

    BPC(bop_read_header(a.ss, /**/ ss_bop, fdname));
    BPC(bop_alloc(/**/ ss_bop));
    BPC(bop_read_values(fdname, /**/ ss_bop));

    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;
    
    BPC(bop_get_n(pp_bop, &n));
    pp = (float*) bop_get_data(pp_bop);
    ss = (float*) bop_get_data(ss_bop);
    
    binning(n, pp, ss,
            a.nx, a.ny, a.nz,
            dx, dy, dz, 0, 0, 0,
            /**/ grid, counts);

    avg(ngrid, counts, dx*dy*dz, /**/ grid);
    
    bov.nx = a.nx; bov.ny = a.ny; bov.nz = a.nz;
    bov.lx = a.lx; bov.ly = a.ly; bov.lz = a.lz;
    bov.ox = bov.oy = bov.oz = 0.f;
    bov.data = grid;
    sprintf(bov.var, "sxx sxy sxz syy syz szz kxx kxy kxz kyy kyz kzz");
    bov.ncmp = NPG;

    bov_alloc(sizeof(float), &bov);

    memcpy(bov.data, grid, ngrid * sizeof(float));

    bov_write_header(a.bov, &bov);
    bov_write_values(a.bov, &bov);
    
    free(grid);
    free(counts);
    
    BPC(bop_fin(pp_bop));
    BPC(bop_fin(ss_bop));
    bov_free(&bov);

    return 0;
}

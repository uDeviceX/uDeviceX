#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

#include "bop_common.h"
#include "bop_serial.h"

#include "bov_common.h"
#include "bov_serial.h"

#include "../common/macros.h"

typedef void (*transform_t)(const float*, const float*, float*);
typedef float (*volume_t) (int, int, int, float, float, float);
    
struct Args {
    float lx, ly, lz;
    float rc[3];
    int nx, ny, nz;
    char **ppp, **ccc, *bov;
    int nin;
    int color;
    transform_t trans;
    volume_t vol;
};

static void usg() {
    fprintf(stderr, "usg: u.p2m.color <color> <c/r> nx ny nz Lx Ly Lz rcx rcy rcz <out> <pp.bop> <cc.bop>\n");
    exit(1);
}

void transform_cart(const float*, const float p0[6], /**/ float p[6]);
void transform_cyl(const float rc[3], const float p0[6], /**/ float p[6]);
float volume_cart(int /*i*/, int /*j*/, int /*k*/, float dx, float dy, float dz);
float volume_cyl(int i, int /*j*/, int /*k*/, float dx, float dy, float dz);

static int shift_args(int *c, char ***v) {
    (*c) --;
    (*v) ++;
    return (*c) > 0;
}

static void parse(int argc, char **argv, /**/ Args *a) {
    enum {X, Y, Z};
    char transfcode;
    int nin;

    // skip executable
    if (!shift_args(&argc, &argv)) usg();
    a->color = atoi(*argv);

    if (!shift_args(&argc, &argv)) usg();
    transfcode = (*argv)[0];

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
    a->rc[X] = atof(*argv);
    if (!shift_args(&argc, &argv)) usg();
    a->rc[Y] = atof(*argv);
    if (!shift_args(&argc, &argv)) usg();
    a->rc[Z] = atof(*argv);

    if (!shift_args(&argc, &argv)) usg();
    a->bov = *argv;


    if (!shift_args(&argc, &argv)) usg();
    a->ppp = argv;

    nin = 0;
    while (strcmp("--", *argv)) {
        if (!shift_args(&argc, &argv)) usg();
        ++nin;
    }
    if (!nin) usg();

    if (!shift_args(&argc, &argv)) usg();
    a->ccc = argv;

    if (argc != nin) usg();

    a->nin = nin;
    
    switch (transfcode) {
    case 'c':
        a->trans = &transform_cart;
        a->vol   = &volume_cart;
        break;
    case 'r':
        a->trans = &transform_cyl;
        a->vol   = &volume_cyl;
        break;
    default:
        fprintf(stderr, "wrong transformation <%c>\n", transfcode);
        exit(1);
    };                
}


/* cartesian coordinates */
void transform_cart(const float*, const float p0[6], /**/ float p[6]) {
    for (int c = 0; c < 6; ++c) p[c] = p0[c];
}

/* cylindrical coordinatess */
void transform_cyl(const float rc[3], const float p0[6], /**/ float p[6]) {
    enum {X, Y, Z, U, V, W};
    float x, y, r, th, costh, sinth;
    x = p0[X] - rc[X];
    y = p0[Y] - rc[Y];

    r = sqrt(x*x + y*y);
    th = atan2(y, x);
    costh = x / r;
    sinth = y / r;

    p[X] = r;
    // printf("[%g %g] [%g %g] %g\n", p0[X], p0[Y], rc[X], rc[Y], r);
    p[Y] = th;
    p[Z] = p0[Z];

    p[U] =  costh * p0[U] + sinth * p0[V];
    p[V] = -sinth * p0[U] + costh * p0[V];
    p[W] = p0[W];
}

float volume_cart(int /*i*/, int /*j*/, int /*k*/, float dx, float dy, float dz) {
    return dx * dy * dz;
}

float volume_cyl(int i, int /*j*/, int /*k*/, float dx, float dy, float dz) {
    float r;
    r = (i + 0.5f) * dx;
    return r * dx * dy * dz;
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

// reduce field quantities in each cell
static void binning(int n, const float *pp, const int *cc, int color,
                    int nx, int ny, int nz,
                    float dx, float dy, float dz,
                    float ox, float oy, float oz,
                    transform_t transform, const float rc[3],
                    /**/ float *grid) {

    int i, cid, c;
    float p[6];
    float *r;
    
    for (i = 0; i < n; ++i) {
        c = cc[i];
        transform(rc, pp + 6 * i, /**/ p);
        r = p;
        cid = r2cid(r, nx, ny, nz, dx, dy, dz, ox, oy, oz);

        if (c   == color)    continue;
        if (cid == INVALID)  continue;
        
        grid[cid] += 1.f;
    }
}

// density: scale by volume of each cell
static void den(int nx, int ny, int nz,
                float dx, float dy, float dz,
                volume_t vol, /**/ float *grid) {
    float s, dV;
    int i, j, k, cid;

    cid = 0;
    for (k = 0; k < nz; ++k) {
        for (j = 0; j < ny; ++j) {
            for (i = 0; i < nx; ++i) {
                dV = vol(i, j, k, dx, dy, dz);
                s = 1.f / dV;

                grid[cid ++] *= s;
            }
        }
    }
}

static void process(Args a, float dx, float dy, float dz, const BopData *pp_bop, const BopData *cc_bop, float *grid) {
    long np;
    const float *pp;
    const int *cc;

    BPC( bop_get_n(pp_bop, &np) );
    pp = (const float*) bop_get_data(pp_bop);
    cc = (const   int*) bop_get_data(cc_bop);
    
    binning(np, pp, cc, a.color,
            a.nx, a.ny, a.nz,
            dx, dy, dz, 0, 0, 0, a.trans, a.rc,
            /**/ grid);    
}

static void read_bop(const char *fname, BopData *bop) {
    char fdname[FILENAME_MAX];
    BPC( bop_read_header(fname, /**/ bop, fdname) );
    BPC( bop_alloc(/**/ bop) );
    BPC( bop_read_values(fdname, /**/ bop) );
}

static void write_bov(Args a, long ngrid, const float *grid) {
    BovData *bov;

    BVC( bov_ini(&bov) );
    BVC( bov_set_gridsize(a.nx, a.ny, a.nz, bov) );
    BVC( bov_set_origin(0, 0, 0, bov) );
    BVC( bov_set_extent(a.lx, a.ly, a.lz, bov) );
    BVC( bov_set_ncomp(1, bov) );
    BVC( bov_set_var("color_density", bov) );
    
    BVC( bov_alloc(bov) );

    memcpy(bov_get_data(bov), grid, ngrid * sizeof(float));

    BVC( bov_write_header(a.bov, bov) );
    BVC( bov_write_values(a.bov, bov) );

    BVC( bov_fin(bov) );
}

int main(int argc, char **argv) {
    Args a;
    BopData *pp_bop, *cc_bop;
    float *grid, dx, dy, dz;
    int i, ngrid;
    size_t sz;
    
    parse(argc, argv, /**/ &a);

    ngrid = a.nx * a.ny * a.nz;
    
    sz = ngrid * sizeof(float);
    grid = (float*) malloc(sz);
    memset(grid, 0, sz);
    
    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;

    for (i = 0; i < a.nin; ++i) {
        BPC( bop_ini(&pp_bop) );
        BPC( bop_ini(&cc_bop) );
        read_bop(a.ppp[i], pp_bop);
        read_bop(a.ccc[i], cc_bop);
        process(a, dx, dy, dz, pp_bop, cc_bop, /**/ grid);
        BPC( bop_fin(pp_bop) );
        BPC( bop_fin(cc_bop) );
    }
        
    den(a.nx, a.ny, a.nz, dx, dy, dz, a.vol, /**/ grid);
    
    write_bov(a, ngrid, grid);
    
    free(grid);

    return 0;
}

/*

  # nTEST: rho.t0
  # set -eu
  # rm -f *out.txt
  # make 
  # t=grid
  # ./binning density c 8 16 6   16 32 12  0 0 0  $t data/test.bop
  # bov2txt $t.bov > rho.out.txt

*/

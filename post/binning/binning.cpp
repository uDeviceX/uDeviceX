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
    char *bop, *bov;
    char *field;
    transform_t trans;
    volume_t vol;
};

static void usg() {
    fprintf(stderr, "usg: u.binning <u/v/w/rho> <c/r> nx ny nz Lx Ly Lz rcx rcy rcz <solvent.bop> <out>\n");
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

    // skip executable
    if (!shift_args(&argc, &argv)) usg();
    a->field = *argv;

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
    a->bop = *argv;
    if (!shift_args(&argc, &argv)) usg();
    a->bov = *argv;

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

static float v2field(const float v[3], char f) {
    enum {X, Y, Z};
    switch(f) {
    case 'u':
        return v[X];
    case 'v':
        return v[Y];
    case 'w':
        return v[Z];
    };
    return 1.f; // default: density -> count
}

// reduce field quantities in each cell
static void binning(int n, const float *pp, char f,
                    int nx, int ny, int nz,
                    float dx, float dy, float dz,
                    float ox, float oy, float oz,
                    transform_t transform, const float rc[3],
                    /**/ float *grid, int *counts) {

    int i, cid;
    float p[6];
    float *r, *u;
    
    for (i = 0; i < n; ++i) {
        transform(rc, pp + 6 * i, /**/ p);
        r = p + 0;
        u = p + 3;
        cid = r2cid(r, nx, ny, nz, dx, dy, dz, ox, oy, oz);

        if (cid != INVALID) {
            counts[cid] ++;
            grid[cid] += v2field(u, f);
        }
    }
}

// average: divide by counts in each cell
static void avg(int n, const int *counts, /**/ float *grid) {
    int i, c;
    float s;
    for (i = 0; i < n; ++i) {
        c = counts[i];
        s = c ? 1.f / c : 1;
        grid[i] *= s;
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

int main(int argc, char **argv) {
    Args a;
    BopData *bop;
    BovData *bov;
    float *grid, dx, dy, dz;
    int ngrid, *counts;
    char fdname[FILENAME_MAX], field;
    size_t sz;
    long np;
    const float *pp;
    
    parse(argc, argv, /**/ &a);

    field = a.field[0];
    ngrid = a.nx * a.ny * a.nz;
    
    sz = ngrid * sizeof(float);
    grid = (float*) malloc(sz);
    memset(grid, 0, sz);
    
    sz = ngrid * sizeof(float);
    counts = (int*) malloc(sz);
    memset(counts, 0, sz);

    BPC( bop_ini(&bop) );
    BVC( bov_ini(&bov) );
    
    BPC( bop_read_header(a.bop, /**/ bop, fdname) );
    BPC( bop_alloc(/**/ bop) );
    BPC( bop_read_values(fdname, /**/ bop) );

    dx = a.lx / a.nx;
    dy = a.ly / a.ny;
    dz = a.lz / a.nz;

    BPC( bop_get_n(bop, &np) );
    pp = (const float*) bop_get_data(bop);
    
    binning(np, pp, field,
            a.nx, a.ny, a.nz,
            dx, dy, dz, 0, 0, 0, a.trans, a.rc,
            /**/ grid, counts);    

    if (field == 'u' ||
        field == 'v' ||
        field == 'w') {
        avg(ngrid, counts, /**/ grid);
    } else { // density
        den(a.nx, a.ny, a.nz, dx, dy, dz, a.vol, /**/ grid);
    }

    
    BVC( bov_set_gridsize(a.nx, a.ny, a.nz, bov) );
    BVC( bov_set_origin(0, 0, 0, bov) );
    BVC( bov_set_extent(a.lx, a.ly, a.lz, bov) );
    BVC( bov_set_ncomp(1, bov) );
    BVC( bov_set_var(a.field, bov) );
    
    BVC( bov_alloc(bov) );

    memcpy(bov_get_data(bov), grid, ngrid * sizeof(float));

    BVC( bov_write_header(a.bov, bov) );
    BVC( bov_write_values(a.bov, bov) );
    
    free(grid);
    free(counts);
    
    BPC( bop_fin(bop) );
    BVC( bov_fin(bov) );

    return 0;
}

/*

  # nTEST: u.t0
  # rm -f *out.txt
  # make 
  # t=grid
  # ./binning u c 8 16 6   16 32 12  0 0 0   data/test.bop $t
  # bov2txt $t.bov > u.out.txt

  # nTEST: rho.t0
  # rm -f *out.txt
  # make 
  # t=grid
  # ./binning density c 8 16 6   16 32 12  0 0 0  data/test.bop $t
  # bov2txt $t.bov > rho.out.txt

  # nTEST: v.rad.t0
  # rm -f *out.txt
  # make 
  # t=grid
  # ./binning v r 16 1 1   0.5 6.29 1  0.5 0.5 0  data/rad.bop $t
  # bov2txt $t.bov > v.out.txt

  # nTEST: rho.rad.t0
  # rm -f *out.txt
  # make 
  # t=grid
  # ./binning density r 16 1 1   0.5 6.29 1  0.5 0.5 0  data/rad.bop $t
  # bov2txt $t.bov > rho.out.txt

*/

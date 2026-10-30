#include "memory.cuh"
#include "macros.cuh"
#include <stdio.h>
#include <vector_types.h>
#include <mpi.h>

struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
struct Coords {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};

namespace m {
int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
}
namespace mpicheck { void check(int code, const char *file, int line); }
#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *file, int line, const char *fmt, ...);

enum {X, Y, Z, D};

void coords_get_view(const Coords *c, Coords_v *v) {
    v->xc = c->xc;
    v->yc = c->yc;
    v->zc = c->zc;

    v->xd = c->xd;
    v->yd = c->yd;
    v->zd = c->zd;

    v->Lx = c->Lx;
    v->Ly = c->Ly;
    v->Lz = c->Lz;
}

void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c0) {
    Coords *c;
    int dims[D], periods[D], coords[D];
    MC(m::Cart_get(cart, D, dims, periods, coords));

    EMALLOC(1, c0);
    c = *c0;

    c->xc = coords[X];
    c->yc = coords[Y];
    c->zc = coords[Z];

    c->xd = dims[X];
    c->yd = dims[Y];
    c->zd = dims[Z];

    c->Lx = Lx;
    c->Ly = Ly;
    c->Lz = Lz;
}

void coords_fin(Coords *c) {
    EFREE(c);
}



int xdomain(const Coords *c) {
    return c->Lx * c->xd;
}

int ydomain(const Coords *c) {
    return c->Ly * c->yd;
}

int zdomain(const Coords *c) {
    return c->Lz * c->zd;
}



float xl2xc(const Coords *c, float xl) {
    return xl - 0.5f * c->Lx * (c->xd - 2 * c->xc - 1);
}

float yl2yc(const Coords *c, float yl) {
    return yl - 0.5f * c->Ly * (c->yd - 2 * c->yc - 1);
}

float zl2zc(const Coords *c, float zl) {
    return zl - 0.5f * c->Lz * (c->zd - 2 * c->zc - 1);
}

void local2center(const Coords *c, float3 rl,      float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}



float xc2xl(const Coords *c, float xc) {
    return xc + 0.5f * c->Lx * (c->xd - 2 * c->xc - 1);
}

float yc2yl(const Coords *c, float yc) {
    return yc + 0.5f * c->Ly * (c->yd - 2 * c->yc - 1);
}

float zc2zl(const Coords *c, float zc) {
    return zc + 0.5f * c->Lz * (c->zd - 2 * c->zc - 1);
}

void center2local(const Coords *c, float3 rc,      float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}



float xl2xg(const Coords *c, float xl) {
    return (c->xc + 0.5f) * c->Lx + xl;
}

float yl2yg(const Coords *c, float yl) {
    return (c->yc + 0.5f) * c->Ly + yl;
}

float zl2zg(const Coords *c, float zl) {
    return (c->zc + 0.5f) * c->Lz + zl;
}

void local2global(const Coords *c, float3 rl,      float3 *rg) {
    rg->x = xl2xg(c, rl.x);
    rg->y = yl2yg(c, rl.y);
    rg->z = zl2zg(c, rl.z);
}



float xg2xl(const Coords *c, float xg) {
    return xg - (c->xc + 0.5f) * c->Lx;
}

float yg2yl(const Coords *c, float yg) {
    return yg - (c->yc + 0.5f) * c->Ly;
}

float zg2zl(const Coords *c, float zg) {
    return zg - (c->zc + 0.5f) * c->Lz;
}

void global2local(const Coords *c, float3 rg,      float3 *rl) {
    rl->x = xg2xl(c, rg.x);
    rl->y = yg2yl(c, rg.y);
    rl->z = zg2zl(c, rg.z);
}



int xlo(const Coords *c) { return xl2xg(c, 0) - 0.5*c->Lx; }
int ylo(const Coords *c) { return yl2yg(c, 0) - 0.5*c->Ly; }
int zlo(const Coords *c) { return zl2zg(c, 0) - 0.5*c->Lz; }

int xhi(const Coords *c) { return xl2xg(c, 0) + 0.5*c->Lx; }
int yhi(const Coords *c) { return yl2yg(c, 0) + 0.5*c->Ly; }
int zhi(const Coords *c) { return zl2zg(c, 0) + 0.5*c->Lz; }



int xs(const Coords *c) { return c->Lx; }
int ys(const Coords *c) { return c->Ly; }
int zs(const Coords *c) { return c->Lz; }
int3 subdomain(const Coords *c) {
    int3 L;
    L.x = xs(c);
    L.y = ys(c);
    L.z = zs(c);
    return L;
}



bool is_end(const Coords *c, int dir) {
    enum {X, Y, Z};
    switch (dir) {
    case X: return c->xc == c->xd - 1;
    case Y: return c->yc == c->yd - 1;
    case Z: return c->zc == c->zd - 1;
    }
    return false;
}

static int bigp(int max, int x, int y, int z) {
    return x > max || y > max || z > max;
}
void coord_stamp(const Coords *c,      char *s) {
    int x, y, z;
    x = c->xc; y = c->yc; z = c->zc;
    if (bigp(999, x, y, z))
        ERR("too big domain [%d %d %d]", x, y, z);
    if (sprintf(s, "%03d.%03d.%03d", x, y, z) < 3)
        ERR("sprintf failed [%d %d %d]", x, y, z);
}

int coords_size(const Coords *c) {
    return c->xd * c->yd * c->zd;
}

struct Config;
void conf_lookup_int3(const Config *, const char *, int3 *);
void coords_ini_conf(MPI_Comm cart, const Config *cfg, Coords **c) {
    int3 L;
    UC(conf_lookup_int3(cfg, "glb/L", &L));
    UC(coords_ini(cart, L.x, L.y, L.z, c));
}

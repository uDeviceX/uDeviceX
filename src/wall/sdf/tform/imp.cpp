#include <stdio.h>
#include <vector_types.h>

#include "utils/error.h"
#include "coords/imp.h"
#include "math/tform/imp.h"

#include "imp.h"

struct TGrid {
    float lo[3], hi[3];
    int n[3];
};

enum {OK, BAD};
static int goodp(const int n[3]) {
    enum {X, Y, Z};
    const int H = 1000000;
    int cl, ch;
    cl = n[X] > 0 && n[Y] > 0 && n[Z] > 0;
    ch = n[X] < H && n[Y] < H && n[Z] < H;
    return (cl && ch) ? OK : BAD;
}


static void grid2grid(TGrid *from, TGrid *to, /**/ Tform *t) {
    tform_grid2grid(from->lo, from->hi, from->n,
                    to->lo, to->hi,   to->n, /**/ t);
}

static void sdf_ini(const Coords *c, const int N[3], /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = t->lo[Y] = t->lo[Z] = 0;
    t->hi[X] = xdomain(c);
    t->hi[Y] = ydomain(c);
    t->hi[Z] = zdomain(c);

    t->n[X] = N[X]; t->n[Y] = N[Y]; t->n[Z] = N[Z];
}

static void tex_ini(const Coords *c, const int T[3], const int M[3], /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(c) - M[X];
    t->lo[Y] = ylo(c) - M[Y];
    t->lo[Z] = zlo(c) - M[Z];
    t->hi[X] = xhi(c) + M[X];
    t->hi[Y] = yhi(c) + M[Y];
    t->hi[Z] = zhi(c) + M[Z];

    t->n[X] = T[X]; t->n[Y] = T[Y]; t->n[Z] = T[Z];
}

static void out_ini(const Coords *c, /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(c);
    t->lo[Y] = ylo(c);
    t->lo[Z] = zlo(c);

    t->hi[X] = xhi(c);
    t->hi[Y] = yhi(c);
    t->hi[Z] = zhi(c);

    t->n[X] = xs(c);
    t->n[Y] = ys(c);
    t->n[Z] = zs(c);
}

static void sub_ini(const Coords *c, /**/ Tform *t) {
    enum {X, Y, Z};
    float a0[3], a1[3], b0[3], b1[3];
    a0[X] = -xs(c)/2; a0[Y] = -ys(c)/2; a0[Z] = -zs(c)/2;
    b0[X] =  xs(c)/2; b0[Y] =  ys(c)/2; b0[Z] =  zs(c)/2;

    a1[X] = xlo(c); a1[Y] = ylo(c); a1[Z] = zlo(c);
    b1[X] = xhi(c); b1[Y] = yhi(c); b1[Z] = zhi(c);
    UC(tform_vector(a0, a1,    b0, b1, /**/ t));
}

void tex2sdf_ini(const Coords *c,
                 const int T[3], const int N[3], const int M[3],
                 /**/ Tform *t) {
    enum {X, Y, Z};
    TGrid tex, sdf;

    if (goodp(T) == BAD) ERR("bad T = [%d %d %d]", T[X], T[Y], T[Z]);
    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    if (goodp(M) == BAD) ERR("bad M = [%d %d %d]", M[X], M[Y], M[Z]);

    tex_ini(c, T, M, /**/ &tex);
    sdf_ini(c, N, /**/ &sdf);

    UC(grid2grid(&tex, &sdf, /**/ t));
}

void out2sdf_ini(const Coords *c, const int N[3], /**/ Tform* t) {
    enum {X, Y, Z};
    TGrid sub, sdf;
    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    out_ini(c, /**/ &sub);
    sdf_ini(c, N, /**/ &sdf);
    UC(grid2grid(&sub, &sdf, /**/ t));
}

void sub2tex_ini(const Coords *c, const int T[3], const int M[3], /**/ Tform *t) {
    enum {X, Y, Z, D};
    float lo[D], hi[D]; 

    if (goodp(T) == BAD) ERR("bad T = [%d %d %d]", T[X], T[Y], T[Z]);
    if (goodp(M) == BAD) ERR("bad M = [%d %d %d]", M[X], M[Y], M[Z]);

    lo[X] = -xs(c)/2 - M[X];
    lo[Y] = -ys(c)/2 - M[Y];
    lo[Z] = -zs(c)/2 - M[Z];

    hi[X] = xs(c)/2 + M[X];
    hi[Y] = ys(c)/2 + M[Y];
    hi[Z] = zs(c)/2 + M[Z];

    tform_to_grid(lo, hi, T, /**/ t);
}


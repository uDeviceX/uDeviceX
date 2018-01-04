#include <stdio.h>
#include <vector_types.h>

#include "utils/error.h"
#include "math/tform/imp.h"

#include "glob/type.h"
#include "glob/imp.h"

#include "imp.h"

struct TGrid {
    float lo[3], hi[3];
    int *n;
};

enum {OK, BAD};
static int goodp(int n[3]) {
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
    t->hi[X] = xdomain(*c);
    t->hi[Y] = ydomain(*c);
    t->hi[Z] = zdomain(*c);
    t->n = N;
}

static void tex_ini(const Coords *c, const int T[3], const int M[3], /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(*c) - M[X];
    t->lo[Y] = ylo(*c) - M[Y];
    t->lo[Z] = zlo(*c) - M[Z];
    t->hi[X] = xhi(*c) + M[X];
    t->hi[Y] = yhi(*c) + M[Y];
    t->hi[Z] = zhi(*c) + M[Z];
    t->n = T;
}

static void sub_ini(const Coords *c, /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(*c);
    t->lo[Y] = ylo(*c);
    t->lo[Z] = zlo(*c);

    t->hi[X] = xhi(*c);
    t->hi[Y] = yhi(*c);
    t->hi[Z] = zhi(*c);

    t->n[X] = xs(*c);
    t->n[Y] = ys(*c);
    t->n[Z] = zs(*c);
}

void tex2sdf_ini(const Coords *c,
                 const int T[3], const int N[3], const int M[3],
                 /**/ Tform *t) {
    enum {X, Y, Z};
    TGrid tex, sdf;

    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    if (goodp(M) == BAD) ERR("bad M = [%d %d %d]", M[X], M[Y], M[Z]);
    if (goodp(T) == BAD) ERR("bad T = [%d %d %d]", T[X], T[Y], T[Z]);

    tex_ini(c, T, M, /**/ &tex);
    sdf_ini(c, N, /**/ &sdf);

    UC(grid2grid(&tex, &sdf, /**/ t));
}

void sub2sdf_ini(const Coords *c, const int N[3], /**/ Tform* t) {
    enum {X, Y, Z};
    TGrid sub, sdf;
    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    sub_ini(c, /**/ &sub);
    sdf_ini(c, N, /**/ &sdf);
    UC(grid2grid(&sub, &sdf, /**/ t));
}

void sub2tex_ini(/**/ Tform*) { ERR("not implimented"); }

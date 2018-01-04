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

static void grid2grid(TGrid *from, TGrid *to, /**/ Tform *t) {
    tform_grid2grid(from->lo, from->hi, from->n,
                    to->lo, to->hi,   to->n, /**/ t);
}

void ini_tex2sdf(const Coords *c,
                 int T[3], int N[3], int M[3],
                 /**/ Tform *t) {
    enum {X, Y, Z};
    TGrid tex, sdf;

    tex.lo[X] = xlo(*c) - M[X];
    tex.lo[Y] = ylo(*c) - M[Y];
    tex.lo[Z] = zlo(*c) - M[Z];
    tex.hi[X] = xhi(*c) + M[X];
    tex.hi[Y] = yhi(*c) + M[Y];
    tex.hi[Z] = zhi(*c) + M[Z];
    tex.n = T;

    sdf.lo[X] = sdf.lo[Y] = sdf.lo[Z] = 0;
    sdf.hi[X] = xdomain(*c);
    sdf.hi[Y] = ydomain(*c);
    sdf.hi[Z] = zdomain(*c);
    sdf.n = N;

    grid2grid(&tex, &sdf, /**/ t);
}

void ini_sub2tex(/**/ Tform*) {

}

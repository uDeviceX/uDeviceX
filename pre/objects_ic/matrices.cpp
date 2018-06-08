#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "type.h"
#include "matrices.h"

enum {X, Y, Z, D, N};

static void identity(real M[N][N]) {
    int i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j)
            M[i][j] = i == j ? 1 : 0;
    }
    M[D][D] = 0;
}

static void apply(const real R[D][D], real M[N][N]) {
    int i, j, k;
    real T[D][D];
    for (i = 0; i < D; ++i) {
        for (j = 0; j < D; ++j) {
            T[i][j] = 0;
            for (k = 0; k < D; ++k)
                T[i][j] += M[i][k] * R[k][j];
        }
    }
    for (i = 0; i < D; ++i) {
        for (j = 0; j < D; ++j)
            M[i][j] = T[i][j];
    }
}

static void rotate_x(real th, real M[N][N]) {
    real c = cos(th), s = sin(th);
    const real R[D][D] = {
        {1., 0., 0.},
        {0.,  c, -s},
        {0.,  s,  c}
    };
    apply(R, M);
}

static void rotate_y(real th, real M[N][N]) {
    real c = cos(th), s = sin(th);
    const real R[D][D] = {
        { c, 0.,  s},
        {0., 1., 0.},
        {-s, 0.,  c}
    };
    apply(R, M);
}

static void rotate_z(real th, real M[N][N]) {
    real c = cos(th), s = sin(th);
    const real R[D][D] = {
        { c, -s, 0.},
        { s,  c, 0.},
        {0., 0., 1.}
    };
    apply(R, M);
}

static void rotate(const Angle *a, real M[N][N]) {
    real thx, thy, thz;
    if (a->type == ANGLE_IN) {
        thx = a->x;
        thy = a->y;
        thz = a->z;
    }
    else {
        thx = drand48() * 2 * M_PI;
        thy = drand48() * 2 * M_PI;
        thz = drand48() * 2 * M_PI;
    }
    rotate_x(thx, M);
    rotate_y(thy, M);
    rotate_z(thz, M);
}

static void scale(real s, real M[N][N]) {
    int i, j;
    for (i = 0; i < D; ++i) {
        for (j = 0; j < D; ++j)
            M[i][j] *= s;
    }
}

static void shift(real x, real y, real z, real M[N][N]) {
    M[X][D] = x;
    M[Y][D] = y;
    M[Z][D] = z;
    M[D][D] = 0;
}

static void row(const real r[N], FILE *f) {
    fprintf(f, "%g %g %g %g\n", r[X], r[Y], r[Z], r[D]);
}

static void dump(const real M[N][N], FILE *f) {
    row(M[X], f);
    row(M[Y], f);
    row(M[Z], f);
    row(M[D], f);
}

void dump_matrices(real sc, const Angle *a, int n, const Particles *p, FILE *stream) {
    int i;
    real x, y, z;
    real M[N][N];

    for (i = 0; i < n; ++i) {
        x = p->xx[i];
        y = p->yy[i];
        z = p->zz[i];

        identity(M);
        rotate(a, M);
        scale(sc, M);
        shift(x, y, z, M);
        dump(M, stream);
    }
}

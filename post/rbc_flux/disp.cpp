#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "type.h"

struct Disp {
    float *xx, *yy, *zz;
    int n;
};

static void zini(int n, float **xx) {
    size_t sz = n * sizeof(float);
    *xx = (float*) malloc(sz);
    memset(*xx, 0, sz);
}

void disp_ini(int n, Disp **disp) {
    Disp *d = (Disp*) malloc(sizeof(Disp));
    *disp = d;
    d->n = n;
    zini(n, &d->xx);
    zini(n, &d->yy);
    zini(n, &d->zz);
}

void disp_fin(Disp *d) {
    free(d->xx);
    free(d->yy);
    free(d->zz);
    free(d);
}


static float disp(int L, float x0, float x1) {
    float dx0, dx;
    dx = dx0 = x1 - x0;
    
    if (fabs(dx) > fabs(dx0 - L)) dx = dx0 - L;
    if (fabs(dx) > fabs(dx0 + L)) dx = dx0 + L;
    return dx;
}

void disp_add(int n, const Com *cc0, const Com *cc1, int L[3], Disp *d) {
    enum {X, Y, Z};
    assert(n == d->n);
    const Com *c0, *c1;
    int i;
    
    for (i = 0; i < n; ++i) {
        c0 = &cc0[i];
        c1 = &cc1[i];
        d->xx[i] += disp(L[X], c0->x, c1->x);
        d->yy[i] += disp(L[Y], c0->y, c1->y);
        d->zz[i] += disp(L[Z], c0->z, c1->z);
    }
}

void disp_reduce(const Disp *d, float tot[3]) {
    enum {X, Y, Z};
    int i;
    tot[X] = tot[Y] = tot[Z] = 0;
    for (i = 0; i < d->n; ++i) {
        tot[X] += d->xx[i];
        tot[Y] += d->yy[i];
        tot[Z] += d->zz[i];
    }
}

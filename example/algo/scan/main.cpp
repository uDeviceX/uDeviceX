#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"
#include "mpi/glb.h"
#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "algo/scan/imp.h"

void set(int *a, int *n) {
    int x, i;
    i = 0;
    while (scanf("%d", &x) == 1)
        a[i++] = x;
    *n = i;
}

#define M 9999

static Scan *w;
static int *x, *y;

static void dump0(int *hst, int n) {
    int i;
    for (i = 0; i < n; i++)
        printf("%d\n", hst[i]);
}

static void dump(int *dev, int n) {
    int hst[M];
    cD2H(hst, dev, n);
    dump0(hst, n);
}

static void scan0(int *hst, int n) { /* local scan wrapper */
    scan_ini(n, &w);

    cH2D(x, hst, n);
    scan_apply(x, n, y,  w);

    scan_fin(w);
}

static void main0() {
    int n, a[M];
    set(a, &n); /* see set.cpp */
    scan0(a, n);
    dump(y, n);
}

static void main1() {
    Dalloc(&x, M);
    Dalloc(&y, M);

    main0();

    Dfree(x);
    Dfree(y);
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    main1();
    m::fin();
}

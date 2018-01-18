#include <stdio.h>
#include <math.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/glb.h" /* mini-MPI and -device */
#include "d/api.h"

#include "inc/dev.h"
#include "utils/cc.h"

#include "algo/scan/imp.h"

/* see set.cpp */
void set(/**/ int*, int*);
#define M 9999

static ScanWork w;
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
    scan_work_ini(n, &w);

    cH2D(x, hst, n);
    scan_apply(x, n, y,  &w);

    scan_work_fin(&w);
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

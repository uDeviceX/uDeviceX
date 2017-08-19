#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "m.h" /* mini-MPI and -device */
#include "d/api.h"

#include "bund.h"
#include "glb.h"

#include "inc/dev.h"
#include "cc.h"

#include "scan/int.h"

#define M 9999
#define N 16
#define value 1

static scan::Work w;
static int *x, *y;

static void dump0(int *hst, int n) {
    int i;
    for (i = 0; i < n; i++)
        printf("%d\n", hst[i]);
}

static void dump(int *dev, int n) {
    int hst[M];
    cD2H0(hst, dev, n);
    dump0(hst, n);
}

static void fill(int e, int lo, int hi, int *dev) {
    int hst[M];
    int i;
    for (i = lo; i < hi; i++) hst[i] = e;
    cH2D0(dev, hst, hi - lo);
}

static void fill0(int e, int *dev) { fill(e, 0, N, dev); }

static void main0() {
    fill0(value, x);
    scan::scan(x, N, y,  &w);
    dump(y, N);
}

static void main1() {
    alloc_work(N, &w);
    Dalloc0(&x, M);
    Dalloc0(&y, M);
    
    main0();

    free_work(&w);
    Dfree0(x);
    Dfree0(y);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main1();
    m::fin();
}

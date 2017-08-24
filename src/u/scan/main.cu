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

/* see set.cpp */
void set(/**/ int*, int*);
#define M 9999

static scan::Work w;
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

static void main0() {
    int n;
    set(y, &n);
    cH2D0(x, y, n);
    scan::scan(x, n, y,  &w);
    dump(y, n);
}

static void main1() {
    alloc_work(M, &w);
    Dalloc(&x, M);
    Dalloc(&y, M);

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

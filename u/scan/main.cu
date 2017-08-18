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

#define N 10
#define value 10

static scan::Work w;
static int *x, *dev;
static int hst[N];

static void dump0() {
    int i;
    for (i = 0; i < N; i++)
        printf("%d\n", hst[i]);
}

static void dump() {
    cD2H0(hst, dev, N);
    dump0();
}

static void main0() {
    Dset(x, value, N);
    scan::scan(x, N, /**/ dev, /*w*/ &w);
    dump();
}

static void main1() {
    alloc_work(N, &w);
    Dalloc0(&x,   N);
    Dalloc0(&dev, N);    
    
    main0();

    free_work(&w);
    Dfree0(x);
    Dfree0(dev);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main1();
    m::fin();
}

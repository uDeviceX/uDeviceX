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

#define N 10
#define value 10

static scan::Work w;
static int *x, *dev;

static void dump0(int *hst) {
    int i;
    for (i = 0; i < N; i++)
        printf("%d\n", hst[i]);
}

static void dump(int *dev) {
    int hst[N];
    cD2H(hst, dev, N);
    dump0(hst);
}

static void main0() {
    Dset(x, value, N);
    scan::scan(x, N, dev,  &w);
    dump(x);
}

static void main1() {
    alloc_work(N, &w);
    Dalloc(&x,   N);
    Dalloc(&dev, N);    
    
    main0();

    free_work(&w);
    Dfree(x);
    Dfree(dev);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main1();
    m::fin();
}

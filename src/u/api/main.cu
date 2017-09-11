#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h"
#include "d/api.h"

#include "glb.h"

#include "inc/dev.h"
#include "utils/cc.h"

#define N 10
#define value 10

static int *dev;

static void dump0(int *hst) {
    int i;
    for (i = 0; i < N; i++) printf("%d\n", hst[i]);
}

static void dump(int *dev) {
    int hst[N];
    cD2H(hst, dev, N);
    dump0(hst);
}

static void fill(int e, int *dev) {
    int hst[N];
    int i;
    for (i = 0; i < N; i++) hst[i] = i;
    cH2D(dev, hst, N);
}

static void main0() {
    fill(value, dev);
    dump(dev);
}

static void main1() {
    Dalloc(&dev, N);
    main0();
    Dfree(dev);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    d::ini();    
    main1();
    m::fin();
}

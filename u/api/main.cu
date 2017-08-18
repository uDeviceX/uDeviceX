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

static int *dev;

static void dump0(int *hst) {
    int i;
    for (i = 0; i < N; i++)
        printf("%d\n", hst[i]);
}

static void dump(int *dev) {
    int hst[N];
    cD2H0(hst, dev, N);
    dump0(hst);
}

static void main0() {
    cudaMemset(dev, 42, (N)*sizeof(*dev));
    dump(dev);
}

static void main1() {
    Dalloc0(&dev, N);
    main0();
    Dfree0(dev);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main1();
    m::fin();
}

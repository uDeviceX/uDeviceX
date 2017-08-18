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

scan::Work w;
int *x, *y;

static void main0() {
    scan::scan(x, N, /**/ y, /*w*/ &w);
    MSG("main0");
}

static void main1() {
    alloc_work(N, &w);
    Dalloc0(&x, N);
    
    main0();

    Dfree0(x);
    free_work(&w);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main1();
    m::fin();
}

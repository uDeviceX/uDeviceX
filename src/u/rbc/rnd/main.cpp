#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/mc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "rbc/rnd/imp.h"

void main0(RbcRnd *rnd, int n) {
    int i;
    float x;
    float *dummpy;
    rbc_rnd_gen(rnd, n, &dummpy);
    for (i = 0; i < n; i++) {
        x = rbc_rnd_get_hst(rnd, i);
        printf("%g\n", x);
    }
}

void main1() {
    int n;
    long seed;
    n = 10;
    seed = ENV;
    RbcRnd *rnd;
    rbc_rnd_ini(&rnd, n, seed);
    main0(rnd, n);
    rbc_rnd_fin(rnd);
}

int main(int argc, char **argv) {
    int rank, size;
    m::ini(&argc, &argv);
    MC(m::Comm_rank(m::cart, &rank));
    MC(m::Comm_size(m::cart, &size));
    msg_ini(rank);
    msg_print("mpi size: %d", size);
    main1();
    m::fin();    
}

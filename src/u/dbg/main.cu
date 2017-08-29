#include <stdio.h>

#include "msg.h"
#include "m.h" /* mini-MPI and -device */
#include "glb.h"

#include "d/api.h"

#include <conf.h>
#include "inc/conf.h"
#include "cc.h"
#include "kl.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "dbg.h"


const int n = 100;
Particle *pp;
Force *ff;

void alloc() {
    CC(d::Malloc((void**) &pp, n * sizeof(Particle)));
    CC(d::Malloc((void**) &ff, n * sizeof(Force)));
}

void free() {
    CC(d::Free(pp));
    CC(d::Free(ff));
}

namespace dev {

__global__ void fill(Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    p.r[0] = p.r[1] = p.r[2] = 0;
    p.v[0] = p.v[1] = p.v[2] = 0;

    if (i >= n) return;
    if (i < 1) p.r[0] = 1.5 * XS;
    pp[i] = p;
}
} // dev

void fill() {
    KL(dev::fill, (k_cnf(n)), (pp, n));
}

void check() {
    dbg::check_pos(pp, n, __FILE__, __LINE__, "pos");
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    alloc();
    fill();
    check();
    free();
    m::fin();
}

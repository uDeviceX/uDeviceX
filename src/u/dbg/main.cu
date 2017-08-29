#include <stdio.h>

#include "msg.h"
#include "m.h" /* mini-MPI and -device */
#include "glb.h"

#include "d/api.h"

#include "inc/conf.h"
#include "cc.h"
#include "kl.h"
#include "inc/type.h"
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

void fill() {}

void check() {}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    alloc();
    
    free();
    m::fin();
}

#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/glb.h"
#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "rbc/rnd/imp.h"

static int    argc;
static char **argv;

/* left shift */
void lshift() {
    argc--;
    if (argc < 1) {
        fprintf(stderr, "u/rbc/rnd: not enough args\n");
        exit(2);
    }
}

void main0(RbcRnd *rnd, int n) {
    int i;
    float x;
    rbc_rnd_gen(rnd, n);
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

void main2() {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi size: %d", m::size);
    main1();
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main2();
}

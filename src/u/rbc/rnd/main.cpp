#include <stdio.h>
#include <stdlib.h>

#include "msg.h"
#include "mpi/glb.h"

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

void main0() {
    rbc::rnd::D *rnd;
    rbc::rnd::ini(rnd);
    rbc::rnd::fin(rnd);
}

void main1() {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    main0();
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main1();
}

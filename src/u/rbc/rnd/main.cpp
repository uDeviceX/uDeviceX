#include <stdio.h>
#include <stdlib.h>

#include "msg.h"
#include "mpi/glb.h"

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
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Hello world!");
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main0();
}

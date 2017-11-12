#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mpi/glb.h"

/* local */
#include "lib/imp.h"

static int    argc;
static char **argv;
/* left shift */
void lshift() {
    argc--;
    if (argc < 1) {
        fprintf(stderr, "u/rbc: not enough args\n");
        exit(2);
    }
}

void main1() {
    const char *cell, *ic;
    cell = argv[argc - 1]; lshift();
    ic   = argv[argc - 1]; lshift();
    m::ini(argc, argv);
    run(cell, ic);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main1();
}

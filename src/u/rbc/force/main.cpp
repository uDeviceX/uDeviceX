#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mpi/glb.h"

/* local */
#include "lib/imp.h"

static int    argc;
static char **argv;

void usg() {
    fprintf(stderr, "usage: ./udx cell.off ic.dat\n");
    fprintf(stderr, "rbc client\n");
    exit(1);
}

/* left shift */
void lshift() {
    argc--;
    if (argc < 1) {
        fprintf(stderr, "u/rbc: not enough args\n");
        exit(2);
    }
}

int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }
void main1() {
    const char *cell, *ic;
    ic   = argv[argc - 1]; lshift();
    cell = argv[argc - 1]; lshift();
    if (eq(cell, "-h") || eq(ic, "-h")) usg();
    m::ini(argc, argv);
    run(cell, ic);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main1();
}

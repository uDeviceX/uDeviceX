#include <stdio.h>

#include "msg.h"
#include "mpi/glb.h"

/* left shift */
void lshift() {
    argc--;
    if (argc < 1) {
        fprintf(stderr, "u/rbc/rnd: not enough args\n");
        exit(2);
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Hello world!");
    m::fin();
}

#include <stdio.h>

#include "msg.h"
#include "m.h" /* mini-MPI and -device */
#include "glb.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Hello world!");
    m::fin();
}

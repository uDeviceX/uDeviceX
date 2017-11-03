#include <stdio.h>

#include "msg.h"
#include "mpi/glb.h"

#include "io/field/h5/imp.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Hello world!");
    m::fin();
}

#include <stdio.h>

#include "utils/msg.h"
#include "mpi/glb.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    MSG("mpi size: %d", m::size);
    MSG("Hello world!");
    m::fin();
}

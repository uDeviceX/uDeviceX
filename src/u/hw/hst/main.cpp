#include <stdio.h>

#include "utils/msg.h"
#include "mpi/glb.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi size: %d", m::size);
    msg_print("Hello world!");
    m::fin();
}

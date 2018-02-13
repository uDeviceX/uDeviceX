#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"

int main(int argc, char **argv) {
    int rank, size;
    m::ini(&argc, &argv);
    MC(m::Comm_rank(m::cart, &rank));
    MC(m::Comm_size(m::cart, &size));
    msg_ini(rank);
    msg_print("mpi size: %d", size);
    msg_print("Hello world!");
    m::fin();
}

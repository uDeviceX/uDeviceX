#include <mpi.h>
#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/msg.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "utils/mc.h"

#include "utils/kl.h"

namespace dev {
#include "dev.h"
}

void main0() {
    KL(dev::main, (1, 1), ());
    dSync();
}

int main(int argc, char **argv) {
    int rank, size, dims[3];
    MPI_Comm cart;
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    msg_print("mpi size: %d", size);
    main0();
    msg_print("Hello world!");

    MC(m::Barrier(cart));
    m::fin();
}

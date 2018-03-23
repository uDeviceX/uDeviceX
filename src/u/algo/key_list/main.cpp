#include <stdio.h>
#include <float.h>
#include <mpi.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"

#include "algo/key_list/imp.h"

void main0() {
    KeyList *q;
    KeyList_ini(&q);
    KeyList_append(q, "x y z");
    
    KeyList_fin(q);
}

int main(int argc, char **argv) {
    int rank, size, dims[3];
    MPI_Comm cart;
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);
    main0();

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    MC(m::Barrier(cart));
    m::fin();
}

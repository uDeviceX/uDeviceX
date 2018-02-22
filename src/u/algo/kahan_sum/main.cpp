#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"

#include "algo/kahan_sum/imp.h"

void main0() {
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    kahan_sum_fin(kahan_sum);
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
    msg_print("mpi size: %d", size);
    msg_print("Hello world!");

    MC(m::Barrier(cart));
    m::fin();
}

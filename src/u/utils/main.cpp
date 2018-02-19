#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"

void main0() {
    int n;
    float *a, *b;

    n = 5;
    EMALLOC(n, &a);
    EMALLOC(n, &b);
    EMEMCPY(n, a, /**/ b);
    EFREE(a);
    EFREE(b);
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
    
    MC(m::Barrier(cart));
    m::fin();
}

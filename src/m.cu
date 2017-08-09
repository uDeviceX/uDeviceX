#include <mpi.h>
#include "common.mpi.h"
#include "l/m.h"

#include "m.h"

namespace m { /* MPI */
const int d = 3;
int periods[d] = {true, true, true};
int rank, size, coords[d], dims[d];
const bool reorder = false;

void ini(int argc, char **argv) {
    m::dims[0] = m::dims[1] = m::dims[2] = 1;
    for (int iarg = 1; iarg < argc && iarg <= 3; iarg++)
    m::dims[iarg - 1] = atoi(argv[iarg]);
    
    MC(MPI_Init(&argc, &argv));
    MC(MPI_Comm_rank(MPI_COMM_WORLD,   &m::rank));
    MC(MPI_Comm_size(MPI_COMM_WORLD,   &m::size));
    MC(MPI_Cart_create(MPI_COMM_WORLD,
                       m::d, m::dims, m::periods, m::reorder,   &l::m::cart));
    MC(MPI_Cart_coords(l::m::cart, m::rank, m::d,   m::coords));
}

}

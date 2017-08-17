#include <stdlib.h>
#include <mpi.h>
#include "inc/mpi.h"
#include "l/m.h"

#include "m.h"

namespace m { /* MPI */
static const int d = 3;
static int periods[d] = {true, true, true};
static const bool reorder = false;

int rank, size, coords[d], dims[d];

void ini(int argc, char **argv) {
    int i;
    dims[0] = dims[1] = dims[2] = 1;
    for (i = 1; i < argc && i <= 3; i++) dims[i - 1] = atoi(argv[i]);
    
    MC(MPI_Init(&argc, &argv));
    MC(MPI_Comm_rank(MPI_COMM_WORLD,   &rank));
    MC(MPI_Comm_size(MPI_COMM_WORLD,   &size));
    MC(MPI_Cart_create(MPI_COMM_WORLD, d, dims, periods, reorder,   &l::m::cart));
    MC(MPI_Cart_coords(l::m::cart, rank, d,   coords));
}

void fin() {
    MC(l::m::Finalize());
}
}

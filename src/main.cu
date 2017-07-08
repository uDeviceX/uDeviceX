#include <mpi.h>
#include <conf.h>
#include "conf.common.h"
#include "m.h" /* MPI */
#include "l/m.h"
#include "common.h"
#include "bund.h"
#include "glb.h"

static void mpi_init(int argc, char **argv) {
    MC(MPI_Init(&argc, &argv));
    MC(MPI_Comm_rank(MPI_COMM_WORLD,   &m::rank));
    MC(MPI_Comm_size(MPI_COMM_WORLD,   &m::size));
    MC(MPI_Cart_create(MPI_COMM_WORLD,
                       m::d, m::dims, m::periods, m::reorder,   &m::cart));
    MC(MPI_Cart_coords(m::cart, m::rank, m::d,   m::coords));
}

int main(int argc, char **argv) {
    m::dims[0] = m::dims[1] = m::dims[2] = 1;
    for (int iarg = 1; iarg < argc && iarg <= 3; iarg++)
    m::dims[iarg - 1] = atoi(argv[iarg]);

    mpi_init(argc, argv);

    // panda specific for multi-gpu testing
    //int device = m::rank % 2 ? 0 : 2;
    int device = 0;
    CC(cudaSetDevice(device));
  
    glb::sim(); /* simulation level globals */
    sim::ini();
    if (RESTART) sim::sim_strt();
    else         sim::sim_gen();
    sim::fin();
  
    MC(l::m::Finalize());
}

#include <mpi.h>
#include <conf.h>
#include "conf.common.h"
#include "m.h" /* MPI */
#include "l/m.h"
#include "common.h"
#include "common.mpi.h"
#include "common.cuda.h"
#include "bund.h"
#include "glb.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);

    // panda specific for multi-gpu testing
    //int device = m::rank % 2 ? 0 : 2;
    int device = 0;
    CC(cudaSetDevice(device));
  
    glb::sim(); /* simulation level globals */
    sim::ini();
    if (RESTART) sim::sim_strt();
    else         sim::sim_gen();
    sim::fin();
  
    m::fin();
}

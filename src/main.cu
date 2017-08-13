#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "m.h"
#include "cc.h"
#include "bund.h"
#include "glb.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("rank: %d", m::rank);

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

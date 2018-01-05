#include <mpi.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "d/api.h"

#include "sim/imp.h"

int main(int argc, char **argv) {
    Sim *sim;

    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi size: %d", m::size);
    d::ini();
    
    sim_ini(argc, argv, m::cart, /**/ &sim);
    if (RESTART) sim_strt(sim);
    else         sim_gen(sim);
    sim_fin(sim);
    m::fin();
    msg_print("end");
}

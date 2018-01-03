#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/glb.h"
#include "d/api.h"

#include "sim/imp.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi size: %d", m::size);

    d::ini();
    sim::ini(argc, argv);
    if (RESTART) sim::sim_strt();
    else         sim::sim_gen();
    sim::fin();
    m::fin();
    msg_print("end");
}

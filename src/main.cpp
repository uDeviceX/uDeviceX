#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h"
#include "d/api.h"

#include "sim/imp.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);

    d::ini();
    sim::ini();
    if (RESTART) sim::sim_strt();
    else         sim::sim_gen();
    sim::fin();
    m::fin();
    MSG("end");
}

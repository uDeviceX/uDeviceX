#include <mpi.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "scheme/time/imp.h"
#include "parser/imp.h"
#include "sim/imp.h"

int main(int argc, char **argv) {
    Sim *sim;
    Time *time;
    Config *cfg;
    float tend;

    m::ini(&argc, &argv);
    msg_ini(m::rank);
    msg_print("mpi rank/size: %d/%d", m::rank, m::size);

    sim_ini(argc, argv, m::cart, /**/ &sim, &time, &tend);
    if (RESTART) sim_strt(sim, time, tend);
    else         sim_gen(sim, time, tend);
    sim_fin(sim, time);

    m::fin();
    msg_print("end");
}

#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "m.h" /* mini-MPI and -device */
#include "d.h"

#include "bund.h"
#include "glb.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("rank: %d", m::rank);

    d::ini();
    glb::sim(); /* simulation level globals */
    sim::ini();
    if (RESTART) sim::sim_strt();
    else         sim::sim_gen();
    sim::fin();
    m::fin();
}

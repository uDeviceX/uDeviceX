#include <mpi.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/error.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "scheme/time/imp.h"
#include "parser/imp.h"
#include "sim/imp.h"

int main(int argc, char **argv) {
    Sim *sim;
    Time *time;
    TimeSeg *time_seg;
    Config *cfg;
    float t0;
    int rank, size;
    
    m::ini(&argc, &argv);
    MC(m::Comm_rank(m::cart, &rank));
    MC(m::Comm_size(m::cart, &size));
    msg_ini(rank);
    msg_print("mpi rank/size: %d/%d", rank, size);
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    t0 = 0;
    UC(time_ini(t0, &time));
    UC(time_seg_ini(cfg, /**/ &time_seg));

    sim_ini(cfg, m::cart, time, /**/ &sim);
    if (RESTART) sim_strt(sim, cfg, time, time_seg);
    else         sim_gen(sim, cfg, time, time_seg);
    sim_fin(sim, time);

    UC(time_seg_fin(time_seg));
    UC(time_fin(time));
    UC(conf_fin(cfg));
    m::fin();
    msg_print("end");
}

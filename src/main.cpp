#include <mpi.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/error.h"
#include "utils/mc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "scheme/time_line/imp.h"
#include "conf/imp.h"
#include "sim/imp.h"

int main(int argc, char **argv) {
    Sim *sim;
    TimeLine *time;
    TimeSeg *time_seg;
    Config *cfg;
    float t0;
    int rank, size, dims[3];
    MPI_Comm cart;
    int restart;
    
    m::ini(&argc, &argv);

    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);
    
    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    msg_print("mpi rank/size: %d/%d", rank, size);
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    t0 = 0;
    UC(time_line_ini(t0, &time));
    UC(time_seg_ini(cfg, /**/ &time_seg));

    sim_ini(cfg, cart, /**/ time, &sim);
    UC(conf_lookup_bool(cfg, "glb.restart", &restart));
    msg_print("read restart: %s", restart ? "YES" : "NO" );
    if (restart) UC(sim_strt(sim, cfg, time, time_seg));
    else         UC(sim_gen(sim, cfg, time, time_seg));
    UC(sim_fin(sim));

    UC(time_seg_fin(time_seg));
    UC(time_line_fin(time));
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
    msg_print("end");
}

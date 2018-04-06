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
    Config *cfg;
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
    
    UC(sim_ini(cfg, cart, &sim));
    UC(conf_lookup_bool(cfg, "glb.restart", &restart));
    msg_print("read restart: %s", restart ? "YES" : "NO" );
    if (restart) UC(sim_strt(sim, cfg));
    else         UC(sim_gen(sim, cfg));
    UC(sim_fin(sim));

    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
    msg_print("end");
}

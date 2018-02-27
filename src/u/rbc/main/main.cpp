#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/error.h"
#include "utils/mc.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "coords/ini.h"
#include "conf/imp.h"
#include "scheme/force/imp.h"
#include "rbc/params/imp.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    int seed;
    Config *cfg;
    Coords *coords;
    BForce *bforce;
    RbcParams *par;
    float dt, te, mass;
    float part_freq;
    int rank, dims[3];
    MPI_Comm cart;

    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    msg_ini(rank);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_float(cfg, "time.dt", &dt));
    UC(conf_lookup_float(cfg, "time.end", &te));
    UC(conf_lookup_int(cfg, "rbc.seed", &seed));
    UC(conf_lookup_float(cfg, "rbc.mass", &mass));

    UC(rbc_params_ini(&par));
    UC(rbc_params_set_conf(cfg, par));

    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(cfg, /**/ bforce));
    UC(conf_lookup_float(cfg, "dump.freq_parts", &part_freq));
    UC(coords_ini_conf(cart, cfg, &coords));
    
    run(cfg, cart, dt, mass, te, seed, coords, part_freq, bforce, "rbc.off", "rbcs-ic.txt", par);
    UC(coords_fin(coords));

    UC(bforce_fin(bforce));

    UC(rbc_params_fin(par));
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}

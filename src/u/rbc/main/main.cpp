#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

#include "utils/msg.h"
#include "utils/error.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "coords/ini.h"
#include "parser/imp.h"
#include "scheme/move/params/imp.h"
#include "scheme/force/imp.h"
#include "rbc/params/imp.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    Config *cfg;
    Coords *coords;
    BForce *bforce;
    MoveParams *moveparams;
    RbcParams *par;
    int part_freq;
    m::ini(&argc, &argv);
    msg_ini(m::rank);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(rbc_params_ini(&par));
    UC(rbc_params_set_conf(cfg, par));

    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(cfg, /**/ bforce));
    UC(conf_lookup_int(cfg, "dump.freq_parts", &part_freq));
    coords_ini(m::cart, &coords);
    
    UC(scheme_move_params_ini(&moveparams));
    UC(scheme_move_params_conf(cfg, /**/moveparams));

    run(coords, part_freq, bforce, moveparams, "rbc.off", "rbcs-ic.txt", par);
    coords_fin(coords);

    UC(bforce_fin(bforce));
    UC(scheme_move_params_fin(moveparams));

    UC(rbc_params_fin(par));
    UC(conf_fin(cfg));
    m::fin();
}

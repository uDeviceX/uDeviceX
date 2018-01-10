#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

#include "utils/msg.h"
#include "utils/error.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "glob/type.h"
#include "glob/ini.h"
#include "parser/imp.h"
#include "scheme/force/imp.h"
#include "scheme/force/conf.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    Config *cfg;
    Coords coords;
    BForce *bforce;

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    
    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(cfg, /**/ bforce));
    coords_ini(m::cart, &coords);
    
    run(coords, bforce, "rbc.off", "rbcs-ic.txt");
    coords_fin(&coords);

    UC(bforce_fin(bforce));

    UC(conf_fin(cfg));
    m::fin();
}

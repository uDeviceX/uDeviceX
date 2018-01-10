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
#include "scheme/force/imp.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    Coords coords;
    BForce *bforce;

    UC(bforce_ini(&bforce));
    UC(bforce_ini_none(bforce));
    coords_ini(m::cart, &coords);
    
    run(coords, bforce, "rbc.off", "rbcs-ic.txt");
    coords_fin(&coords);

    UC(bforce_fin(bforce));
    
    m::fin();
}

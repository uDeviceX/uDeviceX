#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

#include "inc/type.h"

#include "utils/mc.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "conf/imp.h"
#include "coords/ini.h"
#include "coords/imp.h"

#include "struct/partlist/type.h"
#include "clist/imp.h"
#include "inter/color/imp.h"
#include "flu/imp.h"

static void gen(MPI_Comm cart, const Config *cfg) {
    Coords *coords;
    GenColor *gc;
    FluQuants flu;
    int3 L;
    int maxp, numdensity;
    numdensity = 10; // TODO

    UC(coords_ini_conf(cart, cfg, &coords));
    L = subdomain(coords);
    maxp = 2 * numdensity * L.x * L.y * L.z;

    UC(flu_ini(false, false, L, maxp, &flu));
    UC(inter_color_ini(&gc));
    UC(inter_color_set_uniform(gc));
    
    UC(flu_gen_quants(coords, numdensity, gc, &flu));

    UC(inter_color_fin(gc));
    UC(flu_fin(&flu));    
    UC(coords_fin(coords));
}

int main(int argc, char **argv) {
    int rank, size, dims[3];
    MPI_Comm cart;
    Config *cfg;
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    msg_print("mpi size: %d", size);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(gen(cart, cfg));
    
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}

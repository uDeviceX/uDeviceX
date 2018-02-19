#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/ini.h"

#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/mc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "utils/error.h"

#include "io/off/imp.h"
#include "parser/imp.h"


void main0(Config *c) {
    int nv, nt, md;
    MeshRead *cell;
    const char *i, *type; /* input */
    UC(conf_lookup_string(c, "i", &i));
    UC(conf_lookup_string(c, "type", &type));

    if (same_str(type, "off"))
        UC(mesh_read_off(i, &cell));
    else if (same_str(type, "ply"))
        UC(mesh_read_ply(i, &cell));
    else
        ERR("expecting `ply` or `off`: `%s`", type); 

    md = mesh_get_md(cell);
    nv = mesh_get_nv(cell);
    nt = mesh_get_nt(cell);
    printf("%d %d %d\n", nv, nt, md);
    UC(mesh_fin(cell));
}

int main(int argc, char **argv) {
    Config *cfg;
    int rank, dims[3];
    MPI_Comm cart;

    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    msg_ini(rank);

    UC(conf_ini(/**/ &cfg));
    UC(conf_read(argc, argv, /**/ cfg));
    UC(main0(cfg));
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}

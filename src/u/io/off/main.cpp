#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/ini.h"

#include "utils/msg.h"
#include "utils/mc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "utils/error.h"

#include "io/off/imp.h"
#include "parser/imp.h"


void main0(Config *c) {
    int nv, nt, md;
    OffRead *off;
    const char *i; /* input */
    UC(conf_lookup_string(c, "i", &i));
    msg_print("i = '%s'", i);
    UC(off_read(i, &off));

    md = off_get_md(off);
    nv = off_get_nv(off);
    nt = off_get_nt(off);
    msg_print("nv, nt, max degree: %d %d %d", nv, nt, md);
    UC(off_fin(off));
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

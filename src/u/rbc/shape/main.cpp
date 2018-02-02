#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/imp.h"
#include "parser/imp.h"

#include "utils/msg.h"
#include "inc/dev.h"

#include "utils/cc.h"

#include "rbc/adj/imp.h"
#include "rbc/shape/imp.h"

#include "io/off/imp.h"
#include "mpi/wrapper.h"

#include "mpi/glb.h"

void run(OffRead *off) {
    Adj *adj;
    RbcShape *shape;
    int md, nt, nv;
    const int4 *tt;
    const float *rr;
    nt = off_get_nt(off); nv = off_get_nv(off);
    tt = off_get_tri(off); rr = off_get_vert(off);
    md = RBCmd;
    adj_ini(md, nt, nv, tt, /**/ &adj);
    rbc_shape_ini(adj, rr, /**/ &shape);

    //    UC(run1(coords, off, ic, q));

    rbc_shape_fin(shape);
    adj_fin(adj);
}

int main(int argc, char **argv) {
    const char *cell = "rbc.off";
    OffRead *off;
    Config *cfg;
    m::ini(&argc, &argv);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(off_read(cell, /**/ &off));

    run(off);

    UC(off_fin(off));
    UC(conf_fin(cfg));
    m::fin();
}

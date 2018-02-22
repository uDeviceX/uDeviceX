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
#include "utils/mc.h"

#include "rbc/adj/imp.h"
#include "rbc/shape/imp.h"

#include "io/off/imp.h"
#include "mpi/wrapper.h"

#include "mpi/glb.h"

void run(MeshRead *off) {
    Adj *adj;
    RbcShape *shape;
    int md, nt, nv;
    const int4 *tt;
    const float *rr;
    float *A;
    nt = mesh_read_get_nt(off); nv = mesh_read_get_nv(off);
    tt = mesh_read_get_tri(off); rr = mesh_read_get_vert(off);
    md = mesh_read_get_md(off);
    adj_ini(md, nt, nv, tt, /**/ &adj);
    rbc_shape_ini(adj, rr, /**/ &shape);

    rbc_shape_area(shape, /**/ &A);
    //    UC(run1(coords, off, ic, q));

    rbc_shape_fin(shape);
    adj_fin(adj);
}

int main(int argc, char **argv) {
    const char *i; /* input */
    MeshRead *off;
    Config *cfg;
    m::ini(&argc, &argv);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_string(cfg, "i", &i));
    UC(mesh_read_ini_off(i, /**/ &off));

    run(off);

    UC(mesh_read_fin(off));
    UC(conf_fin(cfg));
    m::fin();
}

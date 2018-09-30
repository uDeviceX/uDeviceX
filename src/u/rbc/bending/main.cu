#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "coords/type.h"
#include "coords/ini.h"

#include "utils/imp.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "utils/texo.h"
#include "utils/cc.h"
#include "utils/mc.h"

#include "d/api.h"

#include "rbc/params/imp.h"
#include "rbc/type.h"
#include "rbc/imp.h"
#include "rbc/force/imp.h"
#include "rbc/force/bending/imp.h"

#include "io/mesh_read/imp.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "conf/imp.h"

static void write0(Particle p, Force f0) {
    enum {X, Y, Z};
    float *r, *f;
    r = p.r;
    f = f0.f;
    printf("%g %g %g %g %g %g\n", r[X], r[Y], r[Z], f[X], f[Y], f[Z]);
}

static void write1(int n, Particle *p, Force *f) {
    int i;
    for (i = 0; i < n; i++) write0(p[i], f[i]);
}

static void write(int n, Particle *p, Force *f) {
    Particle *p_hst;
    Force *f_hst;

    EMALLOC(n, &p_hst);
    EMALLOC(n, &f_hst);

    cD2H(p_hst, p, n);
    cD2H(f_hst, f, n);

    write1(n, p_hst, f_hst);

    EFREE(p_hst);
    EFREE(f_hst);
}

int main(int argc, char **argv) {
    Config *cfg;
    Coords *coords;
    RbcParams *par;
    Bending *bending;
    const char *cell, *ic;
    MPI_Comm cart;
    int dims[3];
    MeshRead *off;
    RbcQuants q;
    bool ids = false;
    Force *f;
    float kb, phi;
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(coords_ini_conf(cart, cfg, &coords));
    UC(conf_lookup_string(cfg, "rbc.cell", &cell));
    UC(conf_lookup_string(cfg, "rbc.ic", &ic));
    UC(conf_lookup_float(cfg, "rbc.kb", &kb));
    UC(conf_lookup_float(cfg, "rbc.phi", &phi));

    UC(rbc_params_ini(&par));
    UC(rbc_params_set_bending(kb, phi, /**/ par));

    UC(mesh_read_ini_off(cell, /**/ &off));
    UC(rbc_ini(MAX_CELL_NUM, ids, off, &q));
    UC(rbc_gen_mesh(coords, cart, off, ic, /**/ &q));

    Dalloc(&f, q.n); Dzero(f, q.n);

    UC(bending_juelicher_ini(off, &bending));
    UC(bending_apply(bending, par, &q, f));
    //write(q.n, q.pp, f);
    
    Dfree(f);
    UC(bending_fin(bending));
    rbc_fin(&q);
    mesh_read_fin(off);
    
    UC(rbc_params_fin(par));
    UC(conf_fin(cfg));
    UC(coords_fin(coords));
    MC(m::Barrier(cart));

    dSync();
    m::fin();
}

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
#include "utils/te.h"
#include "utils/texo.h"
#include "utils/cc.h"
#include "utils/mc.h"

#include "d/api.h"

#include "rbc/params/imp.h"
#include "rbc/type.h"
#include "rbc/imp.h"
#include "rbc/force/rnd/imp.h"
#include "rbc/force/imp.h"

#include "io/off/imp.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "parser/imp.h"

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

static void run0(float dt, RbcQuants *q, RbcForce *t, const RbcParams *par, Force *f) {
    rbc_force_apply(t, par, dt, q, /**/ f);
    write(q->n, q->pp, f);
}

static void run1(float dt, RbcQuants *q, RbcForce *t, const RbcParams *par) {
    Force *f;
    Dalloc(&f, q->n);
    Dzero(f, q->n);

    run0(dt, q, t, par, f);
    Dfree(f);
}

static void run2(MPI_Comm cart, float dt, const Coords *coords, MeshRead *off, const char *ic, long seed, const RbcParams *par, RbcQuants *q) {
    RbcForce *t;
    rbc_gen_quants(coords, cart, off, ic, /**/ q);
    rbc_force_ini(off, seed, /**/ &t);
    run1(dt, q, t, par);
    rbc_force_fin(t);
}

void run(MPI_Comm cart, float dt, const Coords *coords, const char *cell, const char *ic, long seed, const RbcParams *par) {
    MeshRead *off;
    RbcQuants q;
    UC(mesh_read_off(cell, /**/ &off));
    UC(rbc_ini(off, &q));
    UC(run2(cart, dt, coords, off, ic, seed, par, &q));
    UC(rbc_fin(&q));
    UC(mesh_fin(off));
}

int main(int argc, char **argv) {
    int seed;
    float dt;
    Config *cfg;
    Coords *coords;
    RbcParams *par;
    const char *cell, *ic;
    MPI_Comm cart;
    int dims[3];
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_float(cfg, "time.dt", &dt));

    UC(coords_ini_conf(cart, cfg, &coords));
    UC(conf_lookup_string(cfg, "rbc.cell", &cell));
    UC(conf_lookup_string(cfg, "rbc.ic", &ic));
    UC(conf_lookup_int(cfg, "rbc.seed", &seed));

    UC(rbc_params_ini(&par));
    UC(rbc_params_set_conf(cfg, par));
    
    UC(run(cart, dt, coords, cell, ic, seed, par));

    UC(rbc_params_fin(par));
    UC(conf_fin(cfg));
    UC(coords_fin(coords));

    MC(m::Barrier(cart));
    m::fin();
}

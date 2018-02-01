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

#include "d/api.h"

#include "rbc/params/imp.h"
#include "rbc/type.h"
#include "rbc/imp.h"
#include "rbc/rnd/imp.h"
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
    EMALOOC(n, &f_hst);

    cD2H(p_hst, p, n);
    cD2H(f_hst, f, n);

    write1(n, p_hst, f_hst);

    EFREE(p_hst);
    EFREE(f_hst);
}

static void run0(RbcQuants q, RbcForce t, const RbcParams *par, Force *f) {
    rbc_force_apply(q, t, par, /**/ f);
    write(q.n, q.pp, f);
}

static void run1(RbcQuants q, RbcForce t, const RbcParams *par) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);

    run0(q, t, par, f);
    Dfree(f);
}

static void run2(OffRead *off, const char *ic, const RbcParams *par, RbcQuants q) {
    Coords *coords;
    RbcForce t;
    coords_ini(m::cart, XS, YS, ZS, &coords);
    rbc_gen_quants(coords, m::cart, off, ic, /**/ &q);
    rbc_force_gen(q, &t);
    run1(q, t, par);
    rbc_force_fin(&t);
    coords_fin(coords);
}

void run(const char *cell, const char *ic, const RbcParams *par) {
    OffRead *off;
    RbcQuants q;
    UC(off_read(cell, /**/ &off));
    UC(rbc_ini(off, &q));
    UC(run2(off, ic, par, q));
    UC(rbc_fin(&q));
    UC(off_fin(off));
}

int main(int argc, char **argv) {
    Config *cfg;
    RbcParams *par;
    const char *cell, *ic;
    m::ini(&argc, &argv);
    conf_ini(&cfg);
    conf_read(argc, argv, cfg);

    conf_lookup_string(cfg, "rbc.cell", &cell);
    conf_lookup_string(cfg, "rbc.ic", &ic);

    rbc_params_ini(&par);
    rbc_params_set_conf(cfg, par);
    
    run(cell, ic, par);

    rbc_params_fin(par);
    conf_fin(cfg);
    m::fin();
}

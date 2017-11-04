#include <assert.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/dev.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "inc/type.h"
#include "d/api.h"

#include "math/dev.h"

#include "imp.h"

namespace meshbb {

/* conf */
enum {MAX_COL = 4};

#include "type.h"
#include "bbstates.h"
#include "dev/roots.h"
#include "dev/utils.h"
#include "dev/intersection.h"
#include "dev/collect.h"
#include "dev/main.h"

void ini(int maxpp, /**/ BBdata *d) {
    CC(d::Malloc((void**) &d->ncols,   maxpp * sizeof(int)));
    CC(d::Malloc((void**) &d->datacol, maxpp * MAX_COL * sizeof(float4)));
    CC(d::Malloc((void**) &d->idcol,   maxpp * MAX_COL * sizeof(int)));
}

void fin(/**/ BBdata *d) {
    CC(d::Free(d->ncols));
    CC(d::Free(d->datacol));
    CC(d::Free(d->idcol));    
}

void reini(int n, /**/ BBdata d) {
    CC(d::MemsetAsync(d.ncols, 0, n * sizeof(int)));
}

void find_collisions(int nm, int nt, int nv, const int4 *tt, const Particle *i_pp, int3 L,
                     const int *starts, const int *counts, const Particle *pp, const Force *ff,
                     /**/ BBdata d) {
    if (!nm) return;
    KL(dev::find_collisions, (k_cnf(nm * nt)),
       (nm, nt, nv, tt, i_pp, L, starts, counts, pp, ff, /**/ d.ncols, d.datacol, d.idcol));
}

void select_collisions(int n, /**/ BBdata d) {
    KL(dev::select_collisions, (k_cnf(n)), (n, /**/ d.ncols, d.datacol, d.idcol));
}


void bounce(int n, BBdata d, const Force *ff, int nt, int nv, const int4 *tt, const Particle *i_pp,
            /**/ Particle *pp, Momentum *mm) {
    KL(dev::perform_collisions, (k_cnf(n)),
       (n, d.ncols, d.datacol, d.idcol, ff, nt, nv, tt, i_pp, /**/ pp, mm));
}


void collect_rig_momentum(int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Solid *ss) {
    KL(dev::collect_rig_mom, (k_cnf(ns * nt)), (ns, nt, nv, tt, pp, mm, /**/ ss));
}

void collect_rbc_momentum(int nc, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Force *ff) {
    KL(dev::collect_rbc_mom, (k_cnf(nc * nt)), (nc, nt, nv, tt, pp, mm, /**/ ff));
}

} // meshbb

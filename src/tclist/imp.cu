#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/def.h"
#include "msg.h"

#include "d/api.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "algo/scan/int.h"

#include "utils/cc.h"
#include "utils/kl.h"

#include "imp.h"

namespace tclist {
#define BBOX_MARGIN 0.1f
#define MAXC (256*256*256)

#include "dev.h"

void ini(int Lx, int Ly, int Lz, int maxtriangles, /**/ TClist *c) {
    int ncells = Lx * Ly * Lz;
    assert(ncells < MAXC);
    
    c->ncells = ncells;
    c->dims = make_int3(Lx, Ly, Lz);

    Dalloc(&c->ss, ncells);
    Dalloc(&c->cc, ncells);
    Dalloc(&c->ii, 27 * maxtriangles);
}

void fin(/**/ TClist *c) {
    Dfree(c->ss);
    Dfree(c->cc);
    Dfree(c->ii);
}


void reini(/**/ TClist *c) {
    size_t sz = c->ncells * sizeof(int);
    CC(d::MemsetAsync(c->cc, 0, sz));
    CC(d::MemsetAsync(c->ss, 0, sz));
}

static void countt(int nt, int nv, const int4 *tt, const int nm, const Particle *pp, /**/ int *cc) {
    if (nm == 0) return;
    KL(dev::countt, (k_cnf(nm*nt)), (nt, tt, nv, pp, nm, /**/ cc));
}

void add_triangles(int nt, int nv, const int4 *tt, const int nm, const Particle *pp, /**/ TClist *c) {
    countt(nt, nv, tt, nm, pp, /**/ c->cc);
}

void scan(/**/ TClist *c, /*w*/ scan::Work *w) {
    scan::scan(c->cc, c->ncells, /**/ c->ss, /*w*/ w);
    CC(d::MemsetAsync(c->cc, 0, c->ncells * sizeof(int)));
}

void fill(int nt, int nv, const int4 *tt, const int nm, const Particle *pp, /**/ TClist *c) {
    if (nm == 0) return;
    KL(dev::fill_ids, (k_cnf(nm*nt)), (nt, tt, nv, pp, nm, c->ss, /**/ c->cc, c->ii));
}

} // tclist

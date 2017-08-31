#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "msg.h"

#include "m.h"
#include "cc.h"
#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "kl.h"

// #define debug_output

#include "mbounce/imp.h"
#include "mbounce/bbstates.h"
#include "mbounce/dbg.h"
#include "mbounce/roots.h"
#include "mbounce/gen.h"
#include "mbounce/gen.intersect.h"
#include "mbounce/gen.tri.h"
#include "mbounce/hst.h"
#include "mbounce/dev.h"

namespace mbounce {

void alloc_ticketM(TicketM *t) {
    CC(cudaMalloc(&t->mm_dev, MAX_PART_NUM * sizeof(Momentum)));
    t->mm_hst = new Momentum[MAX_PART_NUM];
}

void free_ticketM(TicketM *t) {
    CC(cudaFree(t->mm_dev));
    delete[] t->mm_hst;
}

void bounce_hst(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                const int n, const int totnt, /**/ Particle *pp, TicketM *t) {
    sub::dbg::ini_hst();
    
    if (totnt && n) {
        memset(t->mm_hst, 0, totnt * sizeof(Momentum));
        sub::hst::bounce(ff, m.tt, m.nt, m.nv, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, t->mm_hst);
    }
    
    sub::dbg::report_hst();
}

void bounce_dev(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                const int n, const int totnt, /**/ Particle *pp, TicketM *t) {
    sub::dbg::ini_dev();
    
    if (totnt && n) {
        CC(cudaMemsetAsync(t->mm_dev, 0, totnt * sizeof(Momentum)));        
        KL(sub::dev::bounce,
           (k_cnf(n)),
           (ff, m.tt, m.nt, m.nv, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, t->mm_dev));
    }
    
    sub::dbg::report_dev();
}

void bounce_rbc_hst(const Force *ff, const int4 *tt, int nt, int nv, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts,
                    const int *tids, const int n, const int totnt, /**/ Particle *pp, TicketM *t) {
    sub::dbg::ini_hst();
    
    if (totnt && n) {
        memset(t->mm_hst, 0, totnt * sizeof(Momentum));
        sub::hst::bounce(ff, tt, nt, nv, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, t->mm_hst);
    }
    
    sub::dbg::report_hst();
}

void bounce_rbc_dev(const Force *ff, const int4 *tt, int nt, int nv, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts,
                    const int *tids, const int n, const int totnt, /**/ Particle *pp, TicketM *t) {
    sub::dbg::ini_dev();
    
    if (totnt && n) {
        CC(cudaMemsetAsync(t->mm_dev, 0, totnt * sizeof(Momentum)));        
        KL(sub::dev::bounce,
           (k_cnf(n)),
           (ff, tt, nt, nv, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, t->mm_dev));
    }
    
    sub::dbg::report_dev();
}

void collect_rig_hst(int nt, int ns, const TicketM *t, /**/ Solid *ss) {
    int n = ns * nt;
    if (n) sub::hst::collect_rig_mom (t->mm_hst, ns, nt, /**/ ss);
}

void collect_rig_dev(int nt, int ns, const TicketM *t, /**/ Solid *ss) {
    int n = ns * nt;
    KL(sub::dev::collect_rig_mom,
       (k_cnf(n)),
       (t->mm_dev, ns, nt, /**/ ss));
}

} // mbounce

#ifdef debug_output
#undef debug_output
#endif

#include <conf.h> // for dt
#include <cstdio>
#include "common.h"
#include "common.cuda.h"

// #define debug_output

#include "mbounce/bbstates.h"
#include "mbounce/roots.h"
#include "mbounce/gen.h"
#include "mbounce/hst.h"
#include "mbounce/dev.h"
#include "mbounce/imp.h"

namespace mbounce {

void bounce_tcells_hst(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                       const int n, /**/ Particle *pp, Solid *ss) {
#ifdef debug_output
    if (dstep % part_freq == 0)
        for (int c = 0; c < NBBSTATES; ++c) bbstates_hst[c] = 0;
#endif

    if (n) sub::hst::bounce_tcells(ff, m, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, ss);
    
#ifdef debug_output
    if ((++dstep) % part_freq == 0)
        print_states(bbstates_hst);
#endif
}
    
void bounce_tcells_dev(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                       const int n, /**/ Particle *pp, Solid *ss) {
#ifdef debug_output
    if (dstep % part_freq == 0) {
        const int zeros[NBBSTATES] = {0};
        CC(cudaMemcpyToSymbol(bbstates_dev, zeros, NBBSTATES*sizeof(int)));
    }
#endif

    if (n) sub::dev::bounce_tcells <<< k_cnf(n) >>> (ff, m, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, ss);
        
#ifdef debug_output
    if ((++dstep) % part_freq == 0) {
        int bbinfos[NBBSTATES];
        CC(cudaMemcpyFromSymbol(bbinfos, bbstates_dev, NBBSTATES*sizeof(int)));
        print_states(bbinfos);
    }
#endif
}

} // mbounce

#ifdef debug_output
#undef debug_output
#endif

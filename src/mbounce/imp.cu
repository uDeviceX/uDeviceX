#include <conf.h> // for dt
#include <cstdio>
#include "common.h"
#include "common.cuda.h"

#define debug_output

#include "mbounce/bbstates.h"
#include "mbounce/dbg.h"
#include "mbounce/roots.h"
#include "mbounce/gen.h"
#include "mbounce/hst.h"
#include "mbounce/dev.h"
#include "mbounce/imp.h"

namespace mbounce {

void bounce_tcells_hst(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                       const int n, /**/ Particle *pp, Solid *ss) {

    sub::dbg::ini_hst();
    
    if (n) sub::hst::bounce_tcells(ff, m, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, ss);

    sub::dbg::report_hst();
}
    
void bounce_tcells_dev(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                       const int n, /**/ Particle *pp, Solid *ss) {

    sub::dbg::ini_dev();
    
    if (n) sub::dev::bounce_tcells <<< k_cnf(n) >>> (ff, m, i_pp, tcellstarts, tcellcounts, tids, n, /**/ pp, ss);
    
    sub::dbg::report_dev();
}

} // mbounce

#ifdef debug_output
#undef debug_output
#endif

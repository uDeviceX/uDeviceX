#include <stdio.h>
#include <math.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"

#include "d/api.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "algo/scan/int.h"

#include "tcells/int.h"
#include "tcells/imp.h"

namespace tcells {

enum { NCELLS = XS * YS * ZS };

void alloc_quants(int max_num_mesh, /**/ Quants *q) {
    // assume 1 triangle doesn't overlap more than 27 cells
    q->ss_hst = new int[NCELLS];
    q->cc_hst = new int[NCELLS];
    q->ii_hst = new int[27 * MAX_SOLIDS * MAX_FACE_NUM];
    
    Dalloc(&q->ss_dev, NCELLS);
    Dalloc(&q->cc_dev, NCELLS);
    Dalloc(&q->ii_dev, 27 * max_num_mesh * MAX_FACE_NUM);
}

void free_quants(/**/ Quants *q) {
    delete[] q->ss_hst;
    delete[] q->cc_hst;
    delete[] q->ii_hst;

    Dfree0(q->ss_dev);
    Dfree0(q->cc_dev);
    Dfree0(q->ii_dev);
}

void build_hst(const Mesh m, const Particle *i_pp, const int ns, /**/ Quants *q) {
    sub::build_hst(m, i_pp, ns, /**/ q->ss_hst, q->cc_hst, q->ii_hst);
}

void build_dev(const Mesh m, const Particle *i_pp, const int ns, /**/ Quants *q, /*w*/ scan::Work *w) {
    sub::build_dev(m, i_pp, ns, /**/ q->ss_dev, q->cc_dev, q->ii_dev, /*w*/ w);
}

} // tcells

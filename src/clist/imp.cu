#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "m.h"
#include "inc/def.h"
#include "msg.h"
#include "cc.h"
#include "kl.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "scan/int.h"
#include "clist/int.h"

namespace clist {
namespace dev {
#include "clist/dev.h"
}

static void scan(const int *counts, int n, /**/ int *starts) {
    scan::Work ws;
    scan::alloc_work(n, /**/ &ws);
    scan::scan(counts, n, /**/ starts, /*w*/ &ws);
    scan::free_work(&ws);
}

void build(int n, int xcells, int ycells, int zcells,
           /**/ Particle *pp, int *starts, int *counts) {
    if (!n) return;

    int ncells = xcells * ycells * zcells;
    if (!ncells) return;
    int3 cells = make_int3(xcells, ycells, zcells);

    int *ids;
    Particle *ppd;
    Dalloc(&ids, n);
    Dalloc(&ppd, n);

    CC(d::MemsetAsync(counts, 0, ncells * sizeof(int)));
    KL(dev::get_counts, (k_cnf(n)), (pp, n, cells, /**/ counts));
    scan(counts, ncells, /**/ starts);
    DzeroA(counts, ncells);
    KL(dev::get_ids, (k_cnf(n)), (pp, starts, n, cells, /**/ counts, ids));
    KL(dev::gather, (k_cnf(n)), (pp, ids, n, /**/ ppd));

    aD2D(pp, ppd, n);
    Dfree0(ids);
    Dfree0(ppd);
}
}

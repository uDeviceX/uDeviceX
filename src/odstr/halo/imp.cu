#include <stdio.h>
#include <assert.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "inc/def.h"
#include "msg.h"
#include "mpi/glb.h"
#include "utils/cc.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "inc/macro.h"
#include "utils/kl.h"

#include "int.h"         /* local to this dir */

#include "check.h"
#include "common.h"

void halo(const Particle *pp, int n, /**/ int **iidx, int *sizes) {
    CC(d::MemsetAsync(sizes, 0, 27 * sizeof(int)));
    KL(dev::halo, (k_cnf(n)),(pp, n, /**/ iidx, sizes));
}

void alloc_halo_map(HaloMap *h) {
    //    int i;
    //    for (i = 0; i < 27; ++i)
    //        Dalloc(h->iidx_[i], estimate(i));
}

void free_halo_map(HaloMap *h) {
}

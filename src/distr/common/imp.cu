#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "frag/dev.h"
#include "frag/imp.h"

#include "distr/map/type.h"

#include "imp.h"
#include "dev.h"

void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf) {
    enum {THR=128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(nv, THR), nc);

    KL(dev::dcommon_pack_pp_packets, (blck, thrd), (nv, pp, m, /**/ buf));
}

void dcommon_shift_one_frag(int n, const int fid, /**/ Particle *pp) {
    KL(dev::dcommon_shift_one_frag, (k_cnf(n)), (n, fid, /**/ pp));
}

void dcommon_shift_halo(int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp) {
    KL(dev::dcommon_shift_halo, (k_cnf(nhalo)), (starts, /**/ pp));
}

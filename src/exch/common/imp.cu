#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/cc.h"
#include "utils/kl.h"

#include "inc/dev.h"
#include "inc/type.h"
#include "frag/dev.h"
#include "frag/imp.h"

#include "comm/imp.h"
#include "imp.h"
namespace exch_dev {
#include "dev.h"
}

void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf) {
    KL(exch_dev::ecommon_pack_pp, (14 * 16, 128), (pp, ph, /**/ buf));    
}

void ecommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp) {
    KL(exch_dev::ecommon_shift_one_frag, (k_cnf(n)), (L, n, fid, /**/ pp));
}

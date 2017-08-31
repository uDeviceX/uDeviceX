#include <assert.h>
#include <mpi.h>
#include "mpi/wrapper.h"
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "inc/def.h"
#include "msg.h"
#include "m.h"
#include "cc.h"

#include "dual/type.h"
#include "dual/int.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "inc/mpi.type.h"
#include "inc/macro.h"

#include "scan/int.h"
#include "odstr/halo/int.h"

#include "kl.h"

#include "k/read.h"
#include "k/write.h"
#include "k/common.h"

#include "frag.h"

#include "odstr/type.h"
#include "odstr/imp.h"

#include "odstr/imp/mpi.h"
#include "odstr/imp/mpi.ii.h"

#include "odstr/dev/common.h"
#include "dev/check.h"
#include "odstr/dev/utils.h"
#include "odstr/dev/subindex.h"
#include "odstr/dev/shift.h"
#include "odstr/dev/gather.h"

#include "odstr/imp/buf.h"
#include "odstr/imp/ini.h"
#include "odstr/imp/fin.h"

namespace odstr {
namespace sub {

void halo(const Particle *pp, int n, Send *s) { /* see src/odstr/halo */
    halo(pp, n, /**/ s->iidx, s->size_dev);
}

void scan(int n, Send *s) {
    KL(dev::scan, (1, 32), (n, s->size_dev, /**/ s->strt, s->size_pin.DP));
    dSync();
}

void pack_pp(const Particle *pp, int n, Send *s) {
    KL((dev::pack<float2, 3>), (k_cnf(3*n)),((float2*)pp, s->iidx, s->strt, /**/ s->pp.dev));
}

void pack_ii(const int *ii, int n, const Send *s, Pbufs<int>* sii) {
    KL((dev::pack<int, 1>), (k_cnf(n)),(ii, s->iidx, s->strt, /**/ sii->dev));
}


void count(/**/ Recv *r, int *nhalo) {
    int i;
    static int size[27], strt[28];

    size[0] = strt[0] = 0;
    for (i = 1; i < 27; ++i)    size[i] = r->size[i];
    for (i = 1; i < 28; ++i)    strt[i] = strt[i - 1] + size[i - 1];
    CC(cudaMemcpy(r->strt,    strt,    sizeof(strt),    H2D));
    *nhalo = strt[27];
}

int count_sz(Send *s) {
    int i;
    for(i = 0; i < 27; ++i)
        s->size[i] = s->size_pin.D[i];
    return s->size[0]; /* `n' bulk */
}

void unpack_pp(const int n, const Recv *r, /**/ Particle *pp_re) {
    KL((dev::unpack<float2,3>), (k_cnf(3*n)), (r->pp.dev, r->strt, /**/ (float2*) pp_re));
}

void unpack_ii(const int n, const Recv *r, const Pbufs<int> *rii, /**/ int *ii_re) {
    KL((dev::unpack<int,1>), (k_cnf(n)), (rii->dev, r->strt, /**/ ii_re));
}

void subindex(const int n, const Recv *r, /*io*/ Particle *pp_re, int *counts, /**/ uchar4 *subi) {
    KL(dev::shift, (k_cnf(n)), (n, r->strt, /*io*/ (float2*) pp_re));
    KL(dev::subindex, (k_cnf(n)), (n, r->strt, (float2*) pp_re, /*io*/ counts, /**/ subi));
}

void scatter(bool remote, const uchar4 *subi, const int n, const int *start, /**/ uint *iidx) {
    KL(dev::scatter, (k_cnf(n)), (remote, subi, n, start, /**/ iidx));
}

void gather_id(const int *ii_lo, const int *ii_re, int n, const uint *iidx, /**/ int *ii) {
    KL(dev::gather_id, (k_cnf(n)), (ii_lo, ii_re, n, iidx, /**/ ii));
}
void gather_pp(const float2  *pp_lo, const float2 *pp_re, int n, const uint *iidx,
               /**/ float2  *pp, float4  *zip0, ushort4 *zip1) {
    KL(dev::gather_pp, (k_cnf(n)), (pp_lo, pp_re, n,iidx, /**/ pp, zip0, zip1));
}

} // sub
} // odstr

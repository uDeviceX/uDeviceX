#include <mpi.h>
#include "l/m.h"

#include <conf.h>
#include "m.h"

#include "inc/type.h"
#include "common.h"
#include "common.cuda.h"
#include "common.mpi.h"
#include "inc/macro.h"
#include "inc/tmp/pinned.h"

#include "scan/int.h"

#include "conf.common.h"
#include "kl/kl.h"

#include "k/read.h"
#include "k/write.h"
#include "k/common.h"

#include "odstr/type.h"
#include "odstr/imp.h"
#include "odstr/dev.h"
#include "odstr/buf.h"
#include "odstr/ini.h"
#include "odstr/fin.h"

namespace odstr {
namespace sub {

void waitall(MPI_Request *reqs) {
    MPI_Status statuses[26];
    l::m::Waitall(26, reqs, statuses) ;
}

void post_recv(const MPI_Comm cart, const int rank[], const int btc, const int btp,
               MPI_Request *size_req, MPI_Request *mesg_req, Recv *r) {
    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(r->size + i, 1, MPI_INTEGER, rank[i],
                btc + r->tags[i], cart, size_req + c++);

    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(r->pp.hst[i], MAX_PART_NUM, MPI_FLOAT, rank[i],
                btp + r->tags[i], cart, mesg_req + c++);
}

void post_recv_ii(const MPI_Comm cart, const int rank[], const int tags[], const int bt, /**/ MPI_Request *ii_req, Pbufs<int> *rii) {
    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(rii->hst[i], MAX_PART_NUM, MPI_INT, rank[i],
                bt + tags[i], cart, ii_req + c++);
}

void halo(const Particle *pp, int n, Send *s) {
    CC(cudaMemset(s->size_dev, 0,  27*sizeof(s->size_dev[0])));
    KL(dev::halo, (k_cnf(n)),(pp, n, /**/ s->iidx, s->size_dev));
}

void scan(int n, Send *s) {
    KL(dev::scan, (1, 32),(n, s->size_dev, /**/ s->strt, s->size_pin->DP));
    dSync();
}

void pack_pp(const Particle *pp, int n, Send *s) {
    KL((dev::pack<float2, 3>), (k_cnf(3*n)),((float2*)pp, s->iidx, s->strt, /**/ s->pp.dev));
}

void pack_ii(const int *ii, int n, const Send *s, Pbufs<int>* sii) {
    KL((dev::pack<int, 1>), (k_cnf(n)),(ii, s->iidx, s->strt, /**/ sii->dev));
}

int send_sz(MPI_Comm cart, const int rank[], const int btc, /**/ Send *s, MPI_Request *req) {
    for(int i = 0; i < 27; ++i) s->size[i] = s->size_pin->D[i];
    for(int i = 1, cnt = 0; i < 27; ++i)
    l::m::Isend(s->size + i, 1, MPI_INTEGER, rank[i],
                btc + i, cart, &req[cnt++]);
    return s->size[0]; /* `n' bulk */
}

void send_pp(MPI_Comm cart, const int rank[], const int btp, /**/ Send *s, MPI_Request *req) {
    for(int i = 1, cnt = 0; i < 27; ++i)
    l::m::Isend(s->pp.hst[i], s->size[i] * 6, MPI_FLOAT, rank[i],
                btp + i, cart, &req[cnt++]);
}

void send_ii(MPI_Comm cart, const int rank[], const int size[], const int bt, /**/ Pbufs<int> *sii, MPI_Request *req) {
    for(int i = 1, cnt = 0; i < 27; ++i)
    l::m::Isend(sii->hst[i], size[i], MPI_INT, rank[i],
                bt + i, cart, &req[cnt++]);
}

void recv_count(/**/ Recv *r, int *nhalo) {
    int i;
    static int size[27], strt[28];

    size[0] = strt[0] = 0;
    for (i = 1; i < 27; ++i)    size[i] = r->size[i];
    for (i = 1; i < 28; ++i)    strt[i] = strt[i - 1] + size[i - 1];
    CC(cudaMemcpy(r->strt,    strt,    sizeof(strt),    H2D));
    *nhalo = strt[27];
}

void unpack_pp(const int n, const Recv *r, /**/ Particle *pp_re) {
    KL((dev::unpack<float2,3>), (k_cnf(3*n)), (r->pp.dev, r->strt, /**/ (float2*) pp_re));
}

void unpack_ii(const int n, const Recv *r, const Pbufs<int> *rii, /**/ int *ii_re) {
    KL((dev::unpack<int,1>), (k_cnf(n)), (rii->dev, r->strt, /**/ ii_re));
}

void subindex_remote(const int n, const Recv *r, /*io*/ Particle *pp_re, int *counts, /**/ uchar4 *subi) {
    KL(dev::subindex_remote, (k_cnf(n)), (n, r->strt, /*io*/ (float2*) pp_re, counts, /**/ subi));
}

/* TODO: this is not used, why? */
void cancel_recv(/**/ MPI_Request *size_req, MPI_Request *mesg_req) {
    for(int i = 0; i < 26; ++i) l::m::Cancel(size_req + i) ;
    for(int i = 0; i < 26; ++i) l::m::Cancel(mesg_req + i) ;
}

void scatter(bool remote, const uchar4 *subi, const int n, const int *start, /**/ uint *iidx) {
    KL(dev::scatter, (k_cnf(n)),(remote, subi, n, start, /**/ iidx));
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

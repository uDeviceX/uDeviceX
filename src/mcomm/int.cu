#include <mpi.h>
#include <vector>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "common.h"
#include "msg.h"
#include "cc.h"

#include "inc/dev.h"
#include "inc/type.h"
#include "inc/tmp/pinned.h"

#include "minmax.h"

#include "basetags.h"

#include "mcomm/type.h"
#include "mcomm/int.h"
#include "mcomm/imp.h"

namespace mcomm {

void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t) {
    sub::ini_tcom(cart, /**/ &t->cart, t->rnk_ne, t->ank_ne);
    t->first = true;
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(t->first, /**/ &t->cart, &t->sreq, &t->rreq);
}

void alloc_ticketS(TicketS *ts) {
    for (int i = 0; i < 27; ++i) ts->pp_hst[i] = new Particle[MAX_PART_NUM];
    ts->llo = new PinnedHostBuffer2<float3>;
    ts->hhi = new PinnedHostBuffer2<float3>;
}

void free_ticketS(TicketS *ts) {
    for (int i = 0; i < 27; ++i) delete[] ts->pp_hst[i];
    delete ts->llo;
    delete ts->hhi;
}

void alloc_ticketR(const TicketS * ts, TicketR *tr) {
    for (int i = 1; i < 27; ++i) tr->pp_hst[i] = new Particle[MAX_PART_NUM];
    tr->pp_hst[0] = ts->pp_hst[0];
    CC(cudaMalloc(&tr->pp, MAX_PART_NUM * sizeof(Particle)));
}

void free_ticketR(TicketR *tr) {
    for (int i = 1; i < 27; ++i) delete[] tr->pp_hst[i];
    CC(cudaFree(tr->pp));
}

void extents(const Particle *pp, const int nv, const int nm, /**/ TicketS *t) {
    t->llo->resize(nm); t->hhi->resize(nm);
    if (nm) minmax(pp, nv, nm, /**/ t->llo->DP, t->hhi->DP);
    dSync();
}

int map(const int nm, /**/ TicketM *tm, TicketS *ts) {
    return sub::map(ts->llo->D, ts->hhi->D, nm, /**/ tm->travellers, ts->counts);
}

void pack(const Particle *pp, const int nv, const TicketM *tm, /**/ TicketS *ts) {
    sub::pack(pp, nv, tm->travellers, /**/ ts->pp_hst);
}

void post_recv(/**/ TicketCom *tc, TicketR *tr) {
    sub::post_recv(tc->cart, tc->ank_ne, tc->btc, tc->btp, /**/ tr->counts, tr->pp_hst, &tc->rreq);
}

void post_send(int nv, const TicketS *ts, /**/ TicketCom *tc) {
    if (!tc->first) sub::wait_req(&tc->sreq);
    sub::post_send(tc->cart, tc->rnk_ne, tc->btc, tc->btp, nv, ts->counts, ts->pp_hst, /**/ &tc->sreq);
}

void wait_recv(TicketCom *t) {
    sub::wait_req(&t->rreq);
}

int unpack(int nv, int nbulk, /**/ TicketR *tr) {
    tr->counts[0] = nbulk;
    return sub::unpack(tr->counts, tr->pp_hst, nv, /**/ tr->pp);
}

} // mcomm

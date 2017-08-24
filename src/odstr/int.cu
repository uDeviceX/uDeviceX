#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "common.h"
#include "msg.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "cc.h"
#include "dual/type.h"

#include "basetags.h"
#include "l/m.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "inc/mpi.h"

#include "k/read.h"
#include "k/index.h"
#include "kl.h"

#include "rnd/imp.h"
#include "clist/int.h"
#include "flu/int.h"

#include "scan/int.h"

#include "odstr/type.h"
#include "odstr/int.h"
#include "odstr/imp.h"

#include "odstr/com.h"

namespace odstr {

void alloc_ticketD(/*io*/ basetags::TagGen *tg, /**/ TicketD *t) {
    l::m::Comm_dup(l::m::cart, &t->cart);
    sub::ini_comm(t->cart, /**/ t->rank, t->r.tags);
    sub::ini_S(/**/ &t->s);
    sub::ini_R(&t->s, /**/ &t->r);
    t->first = true;
    Dalloc(&t->subi_lo, MAX_PART_NUM);
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketD(/**/ TicketD *t) {
    sub::fin_S(/**/ &t->s);
    sub::fin_R(/**/ &t->r);
    CC(cudaFree(t->subi_lo));
}

void alloc_ticketI(/*io*/ basetags::TagGen *tg, /**/ TicketI *t) {
    t->first = true;
    sub::ini_SRI(/**/ &t->sii, &t->rii);
    t->bt = get_tag(tg);
}

void free_ticketI(/**/ TicketI *t) {
    sub::fin_SRI(/**/ &t->sii, &t->rii);
}

void alloc_ticketU(TicketU *t) {
    Dalloc(&t->subi_re, MAX_PART_NUM);
    Dalloc(&t->iidx, MAX_PART_NUM);
    Dalloc(&t->pp_re, MAX_PART_NUM);
}

void free_ticketU(TicketU *t) {
    CC(cudaFree(t->subi_re));
    CC(cudaFree(t->iidx));
    CC(cudaFree(t->pp_re));
}

void alloc_ticketUI(TicketUI *t) {
    Dalloc(&t->ii_re, MAX_PART_NUM); 
}

void free_ticketUI(TicketUI *t) {
    CC(cudaFree(t->ii_re));
}

void alloc_work(Work *w) {
    scan::alloc_work(XS*YS*ZS, /**/ &w->s);
}

void free_work(Work *w) {
    scan::free_work(/**/ &w->s);
}

void post_recv_ii(const TicketD *td, TicketI *ti) {
    sub::post_recv_ii(td->cart, td->rank, td->r.tags, ti->bt, /**/ ti->recv_ii_req, &ti->rii);
}

void pack_pp(const flu::Quants *q, TicketD *t) {
    if (q->n) {
        sub::halo(q->pp, q->n, /**/ &t->s);
        sub::scan(q->n, /**/ &t->s);
        sub::pack_pp(q->pp, q->n, /**/ &t->s);
        dSync();
    }
}    

void pack_ii(const int n, const flu::QuantsI *q, const TicketD *td, TicketI *ti) {
    if (n) sub::pack_ii(q->ii, n, &td->s, /**/ &ti->sii);
    dSync();
}

void send_ii(const TicketD *td, TicketI *ti) {
    if (!ti->first) sub::waitall_s(ti->send_ii_req);
    ti->first = false;
    sub::send_ii(td->cart, td->rank, td->s.size, ti->bt, /**/ &ti->sii, ti->send_ii_req);
}

void bulk(flu::Quants *q, TicketD *t) {
    int n = q->n, *count = q->cells->count;
    DzeroA(count, XS*YS*ZS);
    KL(k_index::local<false>, (k_cnf(n)),(n, (float2*)q->pp, /*io*/ count, /*o*/ t->subi_lo));
}

void recv_ii(TicketI *t) {
    sub::waitall_r(t->recv_ii_req);
}

void unpack_pp(const TicketD *td, /**/ flu::Quants *q, TicketU *tu, /*w*/ Work *w) {
    const int nhalo = td->nhalo;
    
    int *start = q->cells->start;
    int *count = q->cells->count;
    
    if (nhalo) {
        sub::unpack_pp(nhalo, &td->r, /**/ tu->pp_re);
        sub::subindex(nhalo, &td->r, /*io*/ tu->pp_re, count, /**/ tu->subi_re);
    }

    scan::scan(count, XS*YS*ZS, /**/ start, /*w*/ &w->s);
}

void unpack_ii(const TicketD *td, const TicketI *ti, TicketUI *tui) {
    const int nhalo = td->nhalo;
    if (nhalo) sub::unpack_ii(nhalo, &td->r, &ti->rii, /**/ tui->ii_re);    
}

#define HALO true
#define BULK false

void gather_pp(const TicketD *td, /**/ flu::Quants *q, TicketU *tu, flu::TicketZ *tz) {
    const int nhalo = td->nhalo, nbulk = td->nbulk;
    
    int n = q->n;
    int *start = q->cells->start;
    
    Particle *pp = q->pp, *pp0 = q->pp0;

    sub::scatter(BULK, td->subi_lo,  n, start, /**/ tu->iidx);
    sub::scatter(HALO, tu->subi_re, nhalo, start, /**/ tu->iidx);

    n = nbulk + nhalo;
    
    sub::gather_pp((float2*)pp, (float2*)tu->pp_re, n, tu->iidx,
                   /**/ (float2*)pp0, tz->zip0, tz->zip1);

    q->n = n;

    /* swap */
    q->pp = pp0; q->pp0 = pp; 
}

#undef HALO
#undef BULK


void gather_ii(const int n, const TicketU *tu, const TicketUI *tui , /**/ flu::QuantsI *q) {
    int *ii = q->ii, *ii0 = q->ii0;

    sub::gather_id(ii, tui->ii_re, n, tu->iidx, /**/ ii0);

    /* swap */
    q->ii = ii0; q->ii0 = ii;
}
}

#include <stdio.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/def.h"
#include "msg.h"
#include "m.h"

#include "d/ker.h"
#include "d/api.h"

#include "cc.h"
#include "mpi/wrapper.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "texo.h"
#include "inc/tmp/pinned.h"

#include "basetags.h"

#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"

#include "rdstr/int.h"
#include "rdstr/imp.h"

namespace rdstr {

void alloc_ticketE(/**/ TicketE *t) {
    t->ll = new PinnedHostBuffer2;
    t->hh = new PinnedHostBuffer2;
    t->rr = new float[3 * MAX_CELL_NUM];
}

void free_ticketE(/**/ TicketE *t) {
    delete t->ll;
    delete t->hh;
    delete[] t->rr;
}

void ini_ticketS(/*io*/ basetags::TagGen *tg, /**/ TicketS *t) {
    auto *p = &t->p;
    mdstr::gen::alloc_buf(0, MAX_PART_NUM, /**/ &p->b);
    p->bt = get_tag(tg);
}

void free_ticketS(/**/ TicketS *t) {
    auto *p = &t->p;
    mdstr::gen::free_buf(0, /**/ &p->b);
}

void ini_ticketR(const TicketS *ts, /**/ TicketR *t) {
    auto *p = &t->p;
    auto *ps = &ts->p;
    mdstr::gen::alloc_buf(1, MAX_PART_NUM, /**/ &p->b);
    p->b.dd[0] = ps->b.dd[0];
    p->bt = ps->bt;
}

void free_ticketR(/**/ TicketR *t) {
    auto *p = &t->p;
    mdstr::gen::free_buf(1, /**/ &p->b);
}


void extents(const Particle *pp, int nc, int nv, /**/ TicketE *t) {
    t->ll->resize(nc);
    t->hh->resize(nc);
    sub::extents(pp, nc, nv, /**/ t->ll->DP, t->hh->DP);
}
    
void get_pos(int nc, /**/ TicketE *t) {
    dSync(); // wait for extents
    sub::get_pos(nc, t->ll->D, t->hh->D, /**/ t->rr);
}

void get_reord(int nc, TicketE *te, /**/ TicketP *tp) {
    mdstr::get_reord(te->rr, nc, /**/ tp);
}

void pack(const Particle *pp, int nv, TicketP *tp, /**/ TicketS *ts) {
    sub::pack(tp->reord, tp->scounts, pp, nv, /**/ &ts->p.b);
}

void post_send(int nv, const TicketP *tp, /**/ TicketC *tc, TicketS *ts) {
    if (!tc->first) {
        sub::waitall(/**/ tc->sreqc);
        sub::waitall(/**/ ts->p.req);
    }
    tc->first = false;
    mdstr::post_sendc(tp, /**/ tc);
    auto *pts = &ts->p;
    sub::post_send(nv, tp->scounts, &pts->b, tc->cart, pts->bt, tc->rnk_ne, /**/ pts->req);
}

void post_recv(/**/ TicketP *tp, TicketC *tc, TicketR *tr) {
    mdstr::post_recvc(/**/ tc, tp);
    auto *ptr = &tr->p;
    sub::post_recv(tc->cart, MAX_PART_NUM, ptr->bt, tc->ank_ne, /**/ &ptr->b, ptr->req);
}

void wait_recv(/**/ TicketC *tc, TicketR *tr) {
    mdstr::wait_recvc(/**/ tc);
    sub::waitall(/**/tr->p.req);
}

int unpack(int nv, const TicketR *tr, const TicketP *tp, /**/ Particle *pp) {
    return sub::unpack(nv, &(tr->p.b), tp->rcounts, /**/ pp);
}

void shift(int nv, const TicketP *tp, /**/ Particle *pp) {
    sub::shift(nv, tp->rcounts, /**/ pp);
}

} // rdstr

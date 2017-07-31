#include <cstdio>
#include <mpi.h>

#include "common.h"
#include "common.cuda.h"
#include "common.tmp.h"

#include "basetags.h"

#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"

#include "rdstr/int.h"
#include "rdstr/imp.h"

namespace rdstr {

void alloc_ticketE(/**/ TicketE *t) {
    t->ll = new PinnedHostBuffer2<float3>;
    t->hh = new PinnedHostBuffer2<float3>;
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

void get_reord(TicketE *te, /**/ TicketP *tp) {
    
}

void pack(const Particle *pp, int nv, TicketP *tp, TicketS *ts) {
    sub::pack(tp->reord, tp->scounts, pp, nv, /**/ &ts->p.b);
}

} // rdstr

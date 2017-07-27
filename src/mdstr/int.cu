#include <mpi.h>
#include "m.h"
#include "l/m.h"
#include "basetags.h"

#include "common.h"

#include "mdstr/imp.h"
#include "mdstr/int.h"

namespace mdstr {

void ini_ticketC(/*io*/ basetags::TagGen *tg, /**/ TicketC *t) {
    l::m::Comm_dup(m::cart, &t->cart);
    sub::gen_ne(m::cart, t->rnk_ne, t->ank_ne);
    t->first = true;
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketC(/**/ TicketC *t) {
    if (!t->first) {
        sub::cancelall(t->rreqc);
        sub::cancelall(t->rreqp);
    }
    l::m::Comm_free(&t->cart);
}

void ini_ticketS(/**/ TicketS *t) {
    for (int i = 0; i < 27; ++i) t->pp[i] = new Particle[MAX_PART_NUM];
    for (int i = 0; i < 27; ++i) t->dd[i] = new      int[MAX_CELL_NUM];
    for (int i = 0; i < 27; ++i) t->counts[i] = 0;
}

void free_ticketS(/**/ TicketS *t) {
    for (int i = 0; i < 27; ++i) delete[] t->pp[i];
    for (int i = 0; i < 27; ++i) delete[] t->dd[i];
}

void ini_ticketR(const TicketS *ts, /**/ TicketR *t) {
    t->pp[0] = ts->pp[0]; // bulk
    for (int i = 1; i < 27; ++i) t->pp[i] = new Particle[MAX_PART_NUM];
    for (int i = 0; i < 27; ++i) t->counts[i] = 0;
}

void free_ticketR(/**/ TicketR *t) {
    for (int i = 1; i < 27; ++i) delete[] t->pp[i];
}

void get_dests(const float *rr, int nm, /**/ TicketS *t) {
    sub::get_dests(rr, nm, /**/ t->dd, t->counts);
}

void pack(const Particle *pp, int nv, /**/  TicketS *t) {
    sub::pack(t->dd, t->counts, pp, nv, /**/ t->pp);
}

void post_send(int nv, const TicketS *ts, /**/ TicketC *tc) {
    if (!tc->first) {
        sub::waitall(tc->sreqc);
        sub::waitall(tc->sreqp);
        tc->first = false;
    }
    sub::post_send(nv, ts->counts, ts->pp, tc->cart, tc->btc, tc->btp, tc->rnk_ne, /**/ tc->sreqc, tc->sreqp);
}

void post_recv(const TicketS *ts, /**/ TicketR *tr, TicketC *tc) {
    sub::post_recv(tc->cart, tc->btc, tc->btp, tc->ank_ne, /**/ tr->counts, tr->pp, tc->rreqc, tc->rreqp);
    tr->counts[0] = ts->counts[0]; // bulk
}

void wait_recv(/**/ TicketC *tc) {
    sub::waitall(tc->rreqc);
    sub::waitall(tc->rreqp);
}

int unpack(int nv, const TicketR *t, /**/ Particle *pp) {
    return sub::unpack(nv, t->pp, t->counts, /**/ pp);
}

#undef i2del
} // mdstr

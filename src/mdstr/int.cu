#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"
#include "common.h"
#include "m.h"
#include "l/m.h"

#include "basetags.h"
#include "mdstr/imp.h"
#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"

namespace mdstr {

void ini_ticketC(/*io*/ basetags::TagGen *tg, /**/ TicketC *t) {
    l::m::Comm_dup(l::m::cart, &t->cart);
    sub::gen_ne(l::m::cart, t->rnk_ne, t->ank_ne);
    t->first = true;
    t->btc = get_tag(tg);
}

void free_ticketC(/**/ TicketC *t) {
    l::m::Comm_free(&t->cart);
}

void ini_ticketP(int max_objs, /**/ TicketP *t) {
    for (int i = 0; i < 26; ++i) t->reord[i] = new int[max_objs];
    for (int i = 0; i < 26; ++i) t->scounts[i] = t->rcounts[i] = 0;
}

void free_ticketP(/**/ TicketP *t) {
    for (int i = 0; i < 26; ++i) delete[] t->reord[i];
}

void get_reord(const float *rr, int nm, /**/ TicketP *t) {
    sub::get_reord(rr, nm, /**/ t->reord, t->scounts);
}

void post_sendc(const TicketP *tp, /**/ TicketC *tc) {
    sub::post_sendc(tp->scounts, tc->cart, tc->btc, tc->rnk_ne, /**/ tc->sreqc);
}

void post_recvc(/**/ TicketC *tc, TicketP *tp) {
    tp->rcounts[0] = tp->scounts[0]; /* bulk */
    sub::post_recvc(tc->cart, tc->btc, tc->ank_ne, /**/ tp->rcounts, tc->rreqc);
}

void wait_recvc(/**/ TicketC *tc) {
    sub::waitall(tc->rreqc);
}

#undef i2del
} // mdstr

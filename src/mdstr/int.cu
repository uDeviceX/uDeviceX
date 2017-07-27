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
    l::m::Comm_free(&t->cart);
}

void ini_ticketS(/**/ TicketS *t);
void free_ticketS(/**/ TicketS *t);

void ini_ticketR(const TicketS *ts, /**/ TicketR *t);
void free_ticketR(/**/ TicketR *t);

void pack();
void post();
void wait();
void unpack();

#undef i2del
} // mdstr

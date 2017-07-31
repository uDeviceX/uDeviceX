namespace mdstr {

void ini_ticketC(/*io*/ basetags::TagGen *tg, /**/ TicketC *t);
void free_ticketC(/**/ TicketC *t);

void ini_ticketP(int max_objs, /**/ TicketP *t);
void free_ticketP(/**/ TicketP *t);

void get_reord(const float *rr, int nm, /**/ TicketP *t);

void post_sendc(const TicketP *tp, /**/ TicketC *tc);
void post_recvc(/**/ TicketC *tc, TicketP *tp);

void wait_recvc(/**/ TicketC *tc);

} // mdstr

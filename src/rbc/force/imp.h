namespace rbc { namespace force {
struct TicketT { rbc::rnd::D *rnd; };
void gen_ticket(const Quants q, TicketT *t);
void fin_ticket(TicketT *t);
void apply(const Quants q, const TicketT t, /**/ Force *ff);

}} /* namespace */

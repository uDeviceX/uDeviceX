namespace rbc { namespace force {
struct TicketT { rbc::rnd::D *rnd; };
void gen_ticket(const RbcQuants q, TicketT *t);
void fin_ticket(TicketT *t);
void apply(const RbcQuants q, const TicketT t, /**/ Force *ff);
void stat(/**/ float *pArea, float *pVolume);
}} /* namespace */

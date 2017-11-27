namespace rbc { namespace force {
/* textures ticket */
struct TicketT {
    Texo <float2> texvert;
    rbc::rnd::D *rnd;
};

void gen_ticket(const Quants q, TicketT *t);
void fin_ticket(TicketT *t);
void apply(const Quants q, const TicketT t, /**/ Force *ff);

}} /* namespace */

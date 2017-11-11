namespace rbc { namespace force {
/* textures ticket */
struct TicketT {
    Texo <float2> texvert;
    Texo <int> texadj0, texadj1;
    Texo <int4> textri;
};

void gen_ticket(const Quants q, TicketT *t);
void fin_ticket(TicketT *t);
void forces(const Quants q, const TicketT t, /**/ Force *ff);

}} /* namespace */

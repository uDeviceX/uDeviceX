namespace rdstr {

using mdstr::TicketC;

/* ticket extend */
struct TicketE {
    PinnedHostBuffer2<float3> *ll, *hh;    /* extents of RBCs                   */
    float *rr;                             /* positions used to distribute rbcs */
};

using mdstr::ini_ticketC;
using mdstr::free_ticketC;

using namespace mdstr;

void alloc_ticketE(/**/ TicketE *t);
void free_ticketE(/**/ TicketE *t);

void extents(const Particle *pp, int nc, int nv, /**/ TicketE *t);
void get_pos(int nc, /**/ TicketE *t);
// void get_reord(const float *rr, int nm, /**/ TicketS *t);
// void pack(const Particle *pp, int nv, /**/  TicketS *t);
// void post_send(int nv, const TicketS *ts, /**/ TicketC *tc);
// void post_recv(const TicketS *ts, /**/ TicketR *tr, TicketC *tc);
// void wait_recv(/**/ TicketC *tc);
// int unpack(int nv, const TicketR *t, /**/ Particle *pp);

} // rdstr

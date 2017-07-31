namespace rdstr {

using mdstr::TicketC; /* communication ticket */
using mdstr::TicketP; /* (un)pack ticket      */

struct TicketE {                           /* ticket extents                    */
    PinnedHostBuffer2<float3> *ll, *hh;    /* extents of RBCs                   */
    float *rr;                             /* positions used to distribute rbcs */
};

namespace gen = mdstr::gen;

struct TicketS {
    gen::TicketS <Particle> p;
};

struct TicketR {
    gen::TicketR <Particle> p;
};

using mdstr::ini_ticketC;
using mdstr::free_ticketC;
using mdstr::ini_ticketP;
using mdstr::free_ticketP;

void alloc_ticketE(/**/ TicketE *t);
void free_ticketE(/**/ TicketE *t);

void ini_ticketS(/*io*/ basetags::TagGen *tg, /**/ TicketS *t);
void free_ticketS(/**/ TicketS *t);

void ini_ticketR(const TicketS *ts, /**/ TicketR *t);
void free_ticketR(/**/ TicketR *t);


void extents(const Particle *pp, int nc, int nv, /**/ TicketE *t);
void get_pos(int nc, /**/ TicketE *t);
void get_reord(int nc, TicketE *te, /**/ TicketP *tp);
void pack(const Particle *pp, int nv, TicketP *tp, /**/ TicketS *ts);
void post_send(int nv, const TicketP *tp, /**/ TicketC *tc, TicketS *ts);
void post_recv(/**/ TicketP *tp, TicketC *tc, TicketR *tr);
void wait_recv(/**/ TicketC *tc, TicketR *tr);
int  unpack(int nv, const TicketR *tr, const TicketP *tp, /**/ Particle *pp);
void shift(int nv, const TicketP *tp, /**/ Particle *pp);

} // rdstr

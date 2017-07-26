namespace dpdr {

using sub::int26;
using sub::int27;
using sub::intp26;
using sub::Particlep26;

using sub::Reqs;
using sub::Sbufs;
using sub::Rbufs;
using sub::SIbuf;
using sub::RIbuf;

struct TicketCom {
    /* basetags */
    int btc, btcs, btp;
    MPI_Comm cart;
    Reqs sreq, rreq;
    int recv_tags[26], dstranks[26];
    bool first;
};

struct TicketRnd {
    rnd::KISS *interrank_trunks[26];
    bool interrank_masks[26];
};

struct TicketShalo {
    int26 estimate;
    int ncells;                /* total number of cells in the halo                   */
    int27 fragstarts;          /* cumulative sum of number of cells for each fragment */
    int26 nc;                  /* number of cells per fragment                        */
    int *npdev, *nphst;        /* number of particles on each fragment (pinned)       */
    Sbufs b;
};

struct TicketRhalo {
    int26 estimate;
    int26 nc, np;              /* number of cells, recv sizes */
    Rbufs b;
};

struct TicketICom {
    int bt;
    MPI_Request sreq[26], rreq[26];
    bool first;
};

struct TicketSIhalo {
    SIbuf b;
};

struct TicketRIhalo {
    RIbuf b;
};

void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t);
void free_ticketcom(/**/ TicketCom *t);

void ini_ticketrnd(const TicketCom tc, /**/ TicketRnd *tr);
void free_ticketrnd(/**/ TicketRnd *tr);

void alloc_ticketSh(/**/ TicketShalo *t);
void free_ticketSh(/**/TicketShalo *t);

void alloc_ticketRh(/**/ TicketRhalo *t);
void free_ticketRh(/**/TicketRhalo *t);

void ini_ticketIcom(/*io*/ basetags::TagGen *tg, /**/ TicketICom *t);
void free_ticketIcom(/**/ TicketICom *t);

void alloc_ticketSIh(/**/ TicketSIhalo *t);
void free_ticketSIh(/**/TicketSIhalo *t);

void alloc_ticketRIh(/**/ TicketRIhalo *t);
void free_ticketRIh(/**/TicketRIhalo *t);

/* remote: send functions */

void gather_cells(const int *start, const int *count, /**/ TicketShalo *t);
void copy_cells(/**/ TicketShalo *t);

void pack(const Particle *pp, /**/ TicketShalo *t);
void pack_ii(const int *ii, const TicketShalo *t, /**/ TicketSIhalo *ti);

void post_send(TicketCom *tc, TicketShalo *ts);
void post_send_ii(const TicketCom *tc, const TicketShalo *ts, /**/ TicketICom *tic, TicketSIhalo *tsi);

void post_expected_recv(TicketCom *tc, TicketRhalo *tr);
void post_expected_recv_ii(const TicketCom *tc, const TicketRhalo *tr, /**/ TicketICom *tic, TicketSIhalo *tsi);

void wait_recv(TicketCom *tc);
void wait_recv_ii(TicketICom *tc);

void recv(TicketRhalo *t);
void recv_ii(const TicketRhalo *t, /**/ TicketRIhalo *ti);

void fremote(TicketRnd trnd, TicketShalo ts, TicketRhalo tr, /**/ Force *ff);
}

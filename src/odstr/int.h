namespace odstr {
struct TicketD { /* distribution */
    /* base tags */
    int btc, btp;

    MPI_Comm cart;
    int rank[27];
    MPI_Request send_sz_req[27], recv_sz_req[27];
    MPI_Request send_pp_req[27], recv_pp_req[27];
    bool first = true;
    sub::Send s;
    sub::Recv r;
    uchar4 *subi_lo;           /* local subindices */
    int nhalo, nbulk;
};

struct TicketI { /* int data */
    int bt;                    /* base tag */
    MPI_Request send_ii_req[27], recv_ii_req[27];
    bool first = true;
    sub::Pbufs<int> sii;       /* Send int data    */
    sub::Pbufs<int> rii;       /* Recv int data    */
};

struct TicketU { /* unpack ticket */
    uchar4 *subi_re;           /* remote subindices */
    Particle *pp_re;           /* remote particles  */
    uint *iidx;                /* scatter indices   */
};

struct TicketUI { /* unpack ticket for int data */
    int *ii_re;                /* remote int data    */
};

struct Work {
    scan::Work s;
};

void alloc_ticketD(/*io*/ basetags::TagGen *tg, /**/ TicketD *t);
void free_ticketD(/**/ TicketD *t);

void alloc_ticketI(/*io*/ basetags::TagGen *tg, /**/ TicketI *t);
void free_ticketI(/**/ TicketI *t);

void alloc_ticketU(TicketU *t);
void free_ticketU(TicketU *t);

void alloc_ticketUI(TicketUI *t);
void free_ticketUI(TicketUI *t);

void alloc_work(Work *w);
void free_work(Work *w);

void post_recv_pp(TicketD *t);
void post_recv_ii(const TicketD *td, TicketI *ti);

void pack_pp(const flu::Quants *q, TicketD *t);
void pack_ii(const int n, const flu::QuantsI *q, const TicketD *td, TicketI *ti);

void send_pp(TicketD *t);
void send_ii(const TicketD *td, TicketI *ti);

void bulk(flu::Quants *q, TicketD *t);

void recv_pp(TicketD *t);
void recv_ii(TicketI *t);

void unpack_pp(const TicketD *td, /**/ flu::Quants *q, TicketU *tu, /*w*/ Work *w);
void unpack_ii(const TicketD *td, const TicketI *ti, TicketUI *tui);

void gather_pp(const TicketD *td, /**/ flu::Quants *q, TicketU *tu, flu::TicketZ *tz);
void gather_ii(const int n, const TicketU *tu, const TicketUI *tui , /**/ flu::QuantsI *q);
}

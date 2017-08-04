namespace mcomm {

struct TicketCom { /* communication ticket */
    int btc, btp;           /* basetags           */
    MPI_Comm cart;          /* communicator       */
    Reqs sreq, rreq;        /* requests           */
    int rnk_ne[27];         /* neighbor rank      */
    int ank_ne[27];         /* anti neighbor rank */
    bool first;
};

struct TicketM { /* map ticket : who goes where? */
    std::vector<int> travellers[27];
};

struct TicketS { /* send data */
    Particle *pp_hst[27]; /* particles on host */
    int counts[27];       /* number of meshes  */
    PinnedHostBuffer2<float3> *llo, *hhi; /* extents */
};

struct TicketR { /* recv data */
    Particle *pp_hst[27]; /* particles on host           */
    int counts[27];       /* number of meshes            */
    Particle *pp;         /* particles on dev (unpacked) */
};

void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t);
void free_ticketcom(/**/ TicketCom *t);

void alloc_ticketS(TicketS *ts);
void free_ticketS(TicketS *ts);

void alloc_ticketR(const TicketS * ts, TicketR *tr);
void free_ticketR(TicketR *tr);

void extents(const Particle *pp, const int nv, const int nm, /**/ TicketS *t);
int pack(const Particle *pp, const int nv, const int nm, /**/ TicketS *t);
void post_recv(/**/ TicketCom *tc, TicketR *tr);
void post_send(int nv, const TicketS *ts, /**/ TicketCom *tc);
void wait_recv(TicketCom *t);
int unpack(int nv, int nbulk, /**/ TicketR *tr);

} // mcomm

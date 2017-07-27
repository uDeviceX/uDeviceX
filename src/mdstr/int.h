namespace mdstr {

/* communication ticket (generic part) */
struct TicketC { 
    int btc, btp;                     /* base tags                   */    
    MPI_Comm cart;                    /* Cartesian communicator      */
    MPI_Request sreqc[26], rreqc[26]; /* counts requests             */
    MPI_Request sreqp[26], rreqp[26]; /* particles requests          */
    int rnk_ne[27];                   /* rank      of the neighbor   */
    int ank_ne[27];                   /* anti-rank of the neighbor   */
};

/* send buffers */
struct TicketS {
    int counts[27];                   /* number of leaving particles  */
    Particle *pp[27];                 /* leaving particles            */
};

/* recv buffers */
struct TicketR {
    int counts[27];                   /* number of incoming particles */
    Particle *pp[27];                 /* incoming particles           */
};

void ini_ticketC(/*io*/ basetags::TagGen *tg, /**/ TicketC *t);
void free_ticketC(/**/ TicketC *t);

void ini_ticketS(/**/ TicketS *t);
void free_ticketS(/**/ TicketS *t);

void ini_ticketR(const TicketS *ts, /**/ TicketR *t);
void free_ticketR(/**/ TicketR *t);

void pack();
void post();
void wait();
void unpack();

} // mdstr

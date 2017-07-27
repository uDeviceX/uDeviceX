namespace mdstr {

/* communication ticket (generic part) */
struct TicketC { 
    int btc, btp;                     /* base tags                 */    
    MPI_Comm cart;                    /* Cartesian communicator    */
    MPI_Request sreqc[26], rreqc[26]; /* counts requests           */
    MPI_Request sreqp[26], rreqp[26]; /* particles requests        */
    int rnk_ne[27];                   /* rank      of the neighbor */
    int ank_ne[27];                   /* anti-rank of the neighbor */
};

/* send buffers */
struct TicketS {

};

/* recv buffers */
struct TicketR {
    int counts[27];
};

void pack();
void post();
void wait();
void unpack();

} // mdstr

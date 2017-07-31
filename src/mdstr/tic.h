namespace mdstr {

/* -- common tickets, mandatory for redistribution -- */
struct TicketC {                      /* communication ticket          */
    MPI_Comm cart;                    /* Cartesian communicator        */
    MPI_Request sreqc[26], rreqc[26]; /* counts requests               */
    int btc;                          /* basetag for counts            */
    int rnk_ne[27];                   /* rank      of the neighbor     */
    int ank_ne[27];                   /* anti-rank of the neighbor     */
    bool first;
};

struct TicketP {                      /* packer/unpacker ticket        */
    int scounts[27];                  /* number of leaving objects     */
    int  *reord[27];                  /* which mesh in which fragment? */
    int rcounts[27];                  /* number of incoming objects    */
};

/* -- optional tickets -- */
namespace gen {

template <typename T>
struct TicketS {         /* send ticket       */
    pbuf<T> b;           /* leaving objects   */
    int bt;              /* base tag          */
    MPI_Request req[26]; /* send requests     */    
};

template <typename T>
struct TicketR {         /* recv ticket       */
    pbuf<T> b;           /* incoming objects  */
    int bt;              /* base tag          */
    MPI_Request req[26]; /* recv requests     */    
};

} // gen
} // mdstr

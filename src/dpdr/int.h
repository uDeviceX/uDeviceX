typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

struct TicketCom {
    MPI_Comm cart;
    MPI_Request sendreq[26], sendcellsreq[26], sendcountreq[26];
    MPI_Request recvreq[26], recvcellsreq[26], recvcountreq[26];
    int recv_tags[26], recv_counts[26], dstranks[26];
};

struct Ticketrnd {
    l::rnd::d::KISS *interrank_trunks[26];
    bool interrank_masks[26];
};

struct TicketShalo {
    int estimate[26];
    bool first;
    int ncells;                /* total number of cells in the halo              */
    
    int27 fragstarts;          /* cumulative sum of number of cells for each fragment */

    intp26 str, cnt, cum;      /* see /doc/remote.md                              */
    int26 nc, capacity;        /* number of cells per fragment                    */
    Particlep26 pp;            /* buffer of particles for each fragment           */
    intp26 ii;                 /* buffer of scattered indices for each fragment   */

    /* pinned buffers */
    int *npdev, *nphst;        /* number of particles on each fragment            */
    Particlep26 ppdev, pphst;  /* pinned memory for transfering particles         */
    intp26 cumdev, cumhst;     /* pinned memory for transfering local cum sum     */
};

struct TicketRhalo {
    int estimate[26];
    
    intp26 cum;                /* cellstarts for each fragment (frag coords)      */
    Particlep26 pp;            /* buffer of particles for each fragment           */

    /* pinned buffers */
    Particlep26 ppdev, pphst;  /* pinned memory for transfering particles         */
    intp26 cumdev, cumhst;     /* pinned memory for transfering local cum sum     */
};



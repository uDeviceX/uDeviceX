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

    void setup_frag(const int i, const int est, const int nfragcells) {
        estimate[i] = est;
        CC(cudaMalloc(&str.d[i], (nfragcells + 1) * sizeof(int)));
        CC(cudaMalloc(&cnt.d[i], (nfragcells + 1) * sizeof(int)));
        CC(cudaMalloc(&cum.d[i], (nfragcells + 1) * sizeof(int)));

        CC(cudaMalloc(&ii.d[i], est * sizeof(int)));
        CC(cudaMalloc(&pp.d[i], est * sizeof(Particle)));

        CC(cudaHostAlloc(&pphst.d[i], est * sizeof(Particle), cudaHostAllocMapped));
        CC(cudaHostGetDevicePointer(&ppdev.d[i], pphst.d[i], 0));

        CC(cudaHostAlloc(&cumhst.d[i], est * sizeof(int), cudaHostAllocMapped));
        CC(cudaHostGetDevicePointer(&cumdev.d[i], cumhst.d[i], 0));
    }

    void free_frag(const int i) {
        CC(cudaFree(str.d[i]));
        CC(cudaFree(cnt.d[i]));
        CC(cudaFree(cum.d[i]));

        CC(cudaFree(ii.d[i]));
        CC(cudaFree(pp.d[i]));
        
        CC(cudaFreeHost(cumhst.d[i]));
        CC(cudaFreeHost(pphst.d[i]));
    }
};

struct TicketRhalo {
    int estimate[26];
    
    intp26 cum;                /* cellstarts for each fragment (frag coords)      */
    Particlep26 pp;            /* buffer of particles for each fragment           */

    /* pinned buffers */
    Particlep26 ppdev, pphst;  /* pinned memory for transfering particles         */
    intp26 cumdev, cumhst;     /* pinned memory for transfering local cum sum     */

    void setup_frag(const int i, const int est, const int nfragcells) {
        estimate[i] = est;
        CC(cudaMalloc(&cum.d[i], (nfragcells + 1) * sizeof(int)));
        CC(cudaMalloc(&pp.d[i], est * sizeof(Particle)));

        CC(cudaHostAlloc(&pphst.d[i], est * sizeof(Particle), cudaHostAllocMapped));
        CC(cudaHostGetDevicePointer(&ppdev.d[i], pphst.d[i], 0));

        CC(cudaHostAlloc(&cumhst.d[i], est * sizeof(int), cudaHostAllocMapped));
        CC(cudaHostGetDevicePointer(&cumdev.d[i], cumhst.d[i], 0));
    }

    void free_frag(const int i) {
        CC(cudaFree(cum.d[i]));
        CC(cudaFree(pp.d[i]));
        
        CC(cudaFreeHost(cumhst.d[i]));
        CC(cudaFreeHost(pphst.d[i]));
    }
};

void ini_ticketcom(MPI_Comm cart, /**/ TicketCom *t) {
    sub::ini_tcom(cart, /**/ &t->cart, t->dstranks, t->recv_tags);
}

void fin_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(/**/ &t->cart);
}

void ini_ticketrnd(const TicketCom tc, /**/ Ticketrnd *tr) {
    sub::ini_trnd(tc.dstranks, /**/ tr->interrank_trunks, tr->interrank_masks);
}

void fin_ticketrnd(/**/ Ticketrnd *tr) {
    sub::fin_trnd(/**/ tr->interrank_trunks);
}

void alloc_ticketSh(/**/ TicketShalo *t) {
    
}

void ini_ticketSh(/**/ TicketShalo *t) {
    
}

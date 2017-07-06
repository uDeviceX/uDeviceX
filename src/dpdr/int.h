typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

struct TicketCom {
    MPI_Comm cart;
    sub::Reqs sreq, rreq;
    int recv_tags[26], recv_counts[26], dstranks[26];
    bool first;
};

struct Ticketrnd {
    l::rnd::d::KISS *interrank_trunks[26];
    bool interrank_masks[26];
};

struct TicketShalo {
    int26 estimate;
    int ncells;                /* total number of cells in the halo              */
    
    int27 fragstarts;          /* cumulative sum of number of cells for each fragment */

    intp26 str, cnt, cum;      /* see /doc/remote.md                              */
    int26 nc;                  /* number of cells per fragment                    */
    Particlep26 pp;            /* buffer of particles for each fragment           */
    intp26 ii;                 /* buffer of scattered indices for each fragment   */

    /* pinned buffers */
    int *npdev, *nphst;        /* number of particles on each fragment            */
    Particlep26 ppdev, pphst;  /* pinned memory for transfering particles         */
    intp26 cumdev, cumhst;     /* pinned memory for transfering local cum sum     */

    void alloc_frag(const int i, const int est, const int nfragcells) {
        estimate.d[i] = est;
        nc.d[i] = nfragcells + 1;
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
    int26 nc, np;                  /* recv sizes */

    void alloc_frag(const int i, const int est, const int nfragcells) {
        estimate[i] = est;
        nc.d[i] = nfragcells + 1;
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
    t->first = true;
}

void free_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(t->first, /**/ &t->cart, &t->sreq, &t->rreq);
}

void ini_ticketrnd(const TicketCom tc, /**/ Ticketrnd *tr) {
    sub::ini_trnd(tc.dstranks, /**/ tr->interrank_trunks, tr->interrank_masks);
}

void free_ticketrnd(/**/ Ticketrnd *tr) {
    sub::fin_trnd(/**/ tr->interrank_trunks);
}

/* TODO move this in impl file */
void alloc_tickethalo(/**/ TicketShalo *ts, TicketRhalo *tr) {
    int xsz, ysz, zsz, estimate, nhalocells;

    for (int i = 0; i < 26; ++i) {
        int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
        xsz = d[0] != 0 ? 1 : XS;
        ysz = d[1] != 0 ? 1 : YS;
        zsz = d[2] != 0 ? 1 : ZS;
        nhalocells = xsz * ysz * zsz;

        estimate = numberdensity * HSAFETY_FACTOR * nhalocells;
        estimate = 32 * ((estimate + 31) / 32);

        ts->alloc_frag(i, estimate, nhalocells);
        tr->alloc_frag(i, estimate, nhalocells);
    }

    CC(cudaHostAlloc(&ts->nphst, sizeof(int) * 26, cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&ts->npdev, ts->nphst, 0));

    int s = ts->fragstarts.d[0] = 0;
    for (int i = 0; i < 26; ++i) ts->fragstarts.d[i + 1] = (s += ts->nc.d[i]);
    ts->ncells = s;
}

void free_ticketSh(/**/TicketShalo *t) {
    for (int i = 0; i < 26; ++i) t->free_frag(i);
    CC(cudaFreeHost(t->nphst));
}

void free_ticketRh(/**/TicketRhalo *t) {
    for (int i = 0; i < 26; ++i) t->free_frag(i);
}


/* remote: send functions */

void gather_cells(const int *start, const int *count, /**/ TicketShalo *t) {
    sub::gather_cells(start, count, t->fragstarts, t->nc, t->ncells,
                      /**/ t->str, t->cnt, t->cum);
}

void copy_cells(/**/ TicketShalo *t) {
    sub::copy_cells(t->fragstarts, t->ncells, t->cum, /**/ t->cumdev);
}

void pack(const Particle *pp, /**/ TicketShalo t) {
    sub::pack(t.fragstarts, t.ncells, pp, t.str, t.cnt, t.cum, t.estimate, t.ii, t.pp, t.npdev);
}

void post(TicketCom *tc, TicketShalo *ts) {
    sub::copy_pp(ts->nphst, ts->pp, ts->pphst);
    sub::post(tc->cart, tc->dstranks, ts->nphst, ts->nc, ts->cumhst, ts->pphst,
              /**/ &tc->sreq);
}

void post_expected_recv(TicketCom *tc, TicketRhalo *tr) {
    sub::post_expected_recv(tc->cart, tc->dstranks, tc->recv_tags, tr->estimate, tr->nc,
                       /**/ tr->pphst, tr->np.d, tr->cumhst, &tc->rreq);
}

void wait_recv(TicketCom *tc) {
    sub::wait_req(&tc->rreq);
}

// TODO move this to imp
void recv(TicketRhalo *t) {
    for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(t->pp.d[i], t->pphst.d[i], sizeof(Particle) * t->np.d[i], H2D));
    
    for (int i = 0; i < 26; ++i)
    CC(cudaMemcpyAsync(t->cum.d[i], t->cumhst.d[i],  sizeof(int) * t->nc.d[i], H2D));
}

void fremote(Ticketrnd trnd, TicketShalo ts, TicketRhalo tr, /**/ Force *ff) {
    static BipsBatch::BatchInfo infos[26];

    for (int i = 0; i < 26; ++i) {
        int dx = (i     + 2) % 3 - 1;
        int dy = (i / 3 + 2) % 3 - 1;
        int dz = (i / 9 + 2) % 3 - 1;

        int m0 = 0 == dx;
        int m1 = 0 == dy;
        int m2 = 0 == dz;

        BipsBatch::BatchInfo entry = {
            (float  *)ts.pp.d[i],
            (float2 *)tr.pp.d[i],
            trnd.interrank_trunks[i]->get_float(),
            ts.nphst[i],
            tr.np.d[i],
            trnd.interrank_masks[i],
            tr.cumdev.d[i],
            ts.ii.d[i],
            dx,
            dy,
            dz,
            1 + m0 * (XS - 1),
            1 + m1 * (YS - 1),
            1 + m2 * (ZS - 1),
            (BipsBatch::HaloType)(abs(dx) + abs(dy) + abs(dz))};

        infos[i] = entry;
    }

    BipsBatch::interactions(infos, (float *)ff);
}

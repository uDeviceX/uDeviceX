using sub::int26;
using sub::int27;
using sub::intp26;
using sub::Particlep26;

struct TicketCom {
    /* basetags */
    int btc, btcs, btp;
    MPI_Comm cart;
    sub::Reqs sreq, rreq;
    int recv_tags[26], recv_counts[26], dstranks[26];
    bool first;
};

struct TicketRnd {
    l::rnd::d::KISS *interrank_trunks[26];
    bool interrank_masks[26];
};

struct TicketShalo {
    int26 estimate;
    int ncells;                /* total number of cells in the halo                   */
    int27 fragstarts;          /* cumulative sum of number of cells for each fragment */
    int26 nc;                  /* number of cells per fragment                        */
    int *npdev, *nphst;        /* number of particles on each fragment (pinned)       */
    sub::Sbufs b;
};

struct TicketRhalo {
    int26 estimate;
    int26 nc, np;              /* number of cells, recv sizes */
    sub::Rbufs b;
};

struct TicketICom {
    int bt;
    MPI_Request sreq[26], rreq[26];
    bool first;
};

struct TicketSIhalo {
    sub::SIbuf b;
};

struct TicketRIhalo {
    sub::RIbuf b;
};

void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t) {
    sub::ini_tcom(cart, /**/ &t->cart, t->dstranks, t->recv_tags); 
    t->first = true;
    t->btc  = get_tag(tg);
    t->btcs = get_tag(tg);
    t->btp  = get_tag(tg);
}

void free_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(t->first, /**/ &t->cart, &t->sreq, &t->rreq);
}

void ini_ticketrnd(const TicketCom tc, /**/ TicketRnd *tr) {
    sub::ini_trnd(tc.dstranks, /**/ tr->interrank_trunks, tr->interrank_masks);
}

void free_ticketrnd(/**/ TicketRnd *tr) {
    sub::fin_trnd(/**/ tr->interrank_trunks);
}

void alloc_ticketSh(/**/ TicketShalo *t) {
    sub::ini_ticketSh(/**/ &t->b, &t->estimate, &t->nc);

    CC(cudaHostAlloc(&t->nphst, sizeof(int) * 26, cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&t->npdev, t->nphst, 0));

    int s = t->fragstarts.d[0] = 0;
    for (int i = 0; i < 26; ++i) t->fragstarts.d[i + 1] = (s += t->nc.d[i]);
    t->ncells = s;
}

void free_ticketSh(/**/TicketShalo *t) {
    sub::free_Sbufs(/**/ &t->b);
    CC(cudaFreeHost(t->nphst));
}

void alloc_ticketRh(/**/ TicketRhalo *t) {
    sub::ini_ticketRh(/**/ &t->b, &t->estimate, &t->nc);
}

void free_ticketRh(/**/TicketRhalo *t) {
    sub::free_Rbufs(/**/ &t->b);
}

void ini_ticketIcom(/*io*/ basetags::TagGen *tg, /**/ TicketICom *t) {    
    t->first = true;
    t->bt    = get_tag(tg);    
}

void free_ticketIcom(/**/ TicketCom *t) {t->first = true;}

void alloc_ticketSIh(/**/ TicketSIhalo *t) {
    sub::ini_ticketSIh(/**/ &t->b);
}

void free_ticketSIh(/**/TicketSIhalo *t) {
    sub::free_SIbuf(/**/ &t->b);
}

void alloc_ticketRIh(/**/ TicketRIhalo *t) {
    sub::ini_ticketRIh(/**/ &t->b);
}

void free_ticketRIh(/**/TicketRIhalo *t) {
    sub::free_RIbuf(/**/ &t->b);
}



/* remote: send functions */

void gather_cells(const int *start, const int *count, /**/ TicketShalo *t) {
    sub::gather_cells(start, count, t->fragstarts, t->nc, t->ncells,
                      /**/ t->b.str, t->b.cnt, t->b.cum);
}

void copy_cells(/**/ TicketShalo *t) {
    sub::copy_cells(t->fragstarts, t->ncells, t->b.cum, /**/ t->b.cumdev);
}

void pack(const Particle *pp, /**/ TicketShalo *t) {
    sub::pack(t->fragstarts, t->ncells, pp, t->b.str, t->b.cnt, t->b.cum, t->estimate, /**/ t->b.ii, t->b.pp, t->npdev);
}

void pack_ii(const int *ii, const TicketShalo *t, /**/ TicketSIhalo *ti) {
    sub::pack_ii(t->fragstarts, t->ncells, ii, t->b.str, t->b.cnt, t->b.cum, t->estimate, /**/ ti->b.ii);
}

void post_send(TicketCom *tc, TicketShalo *ts) {
    if (!tc->first) sub::wait_req(&tc->sreq);
    sub::copy_pp(ts->nphst, ts->b.pp, /**/ ts->b.pphst);
    sub::post_send(tc->cart, tc->dstranks, ts->nphst, ts->nc, ts->b.cumhst, ts->b.pphst,
              tc->btcs, tc->btc, tc->btp, /**/ &tc->sreq);
}

void post_expected_recv(TicketCom *tc, TicketRhalo *tr) {
    sub::post_expected_recv(tc->cart, tc->dstranks, tc->recv_tags, tr->estimate.d, tr->nc,
                            tc->btcs, tc->btc, tc->btp, /**/ tr->b.pphst, tr->np.d, tr->b.cumhst, &tc->rreq);
}

void wait_recv(TicketCom *tc) {
    sub::wait_req(&tc->rreq);
}

void recv(TicketRhalo *t) {
    sub::recv(t->np.d, t->nc.d, /**/ &t->b);
}

// TODO move this to imp
void fremote(TicketRnd trnd, TicketShalo ts, TicketRhalo tr, /**/ Force *ff) {
    int i;
    int dx, dy, dz;
    int m0, m1, m2;
    SFrag sfrag[26];
    Frag   frag[26];
    Rnd     rnd[26];

    for (i = 0; i < 26; ++i) {
        dx = (i     + 2) % 3 - 1;
        dy = (i / 3 + 2) % 3 - 1;
        dz = (i / 9 + 2) % 3 - 1;

        m0 = 0 == dx;
        m1 = 0 == dy;
        m2 = 0 == dz;

        sfrag[i] = {
            (float*)ts.b.pp.d[i],
            ts.b.ii.d[i],
            ts.nphst[i]};

        frag[i] = {
            (float2*)tr.b.pp.d[i],
            tr.b.cumdev.d[i],
            dx,
            dy,
            dz,
            1 + m0 * (XS - 1),
            1 + m1 * (YS - 1),
            1 + m2 * (ZS - 1),
            (FragType)(abs(dx) + abs(dy) + abs(dz))};

        rnd[i] = {trnd.interrank_trunks[i]->get_float(), trnd.interrank_masks[i]};
    }

    bipsbatch::interactions(sfrag, frag, rnd, (float*)ff);
}

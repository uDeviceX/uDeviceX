struct TicketCom { /* communication ticket */
    int btc, btp;           /* basetags     */
    MPI_Comm cart;          /* communicator */
    sub::Reqs sreq, rreq;   /* requests     */
    int recv_tags[26], dstranks[26];
    bool first;
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

void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t) {
    sub::ini_tcom(cart, /**/ &t->cart, t->dstranks, t->recv_tags);
    t->first = true;
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(t->first, /**/ &t->cart, &t->sreq, &t->rreq);
}

void alloc_ticketS(TicketS *ts) {
    for (int i = 0; i < 27; ++i) ts->pp_hst[i] = new Particle[MAX_PART_NUM];
    ts->llo = new PinnedHostBuffer2<float3>;
    ts->hhi = new PinnedHostBuffer2<float3>;
}

void free_ticketS(TicketS *ts) {
    for (int i = 0; i < 27; ++i) delete[] ts->pp_hst[i];
    delete ts->llo;
    delete ts->hhi;
}

void alloc_ticketR(const TicketS * ts, TicketR *tr) {
    for (int i = 1; i < 27; ++i) tr->pp_hst[i] = new Particle[MAX_PART_NUM];
    tr->pp_hst[0] = ts->pp_hst[0];
    CC(cudaMalloc(&tr->pp, MAX_PART_NUM * sizeof(Particle)));
}

void free_ticketR(TicketR *tr) {
    for (int i = 1; i < 27; ++i) delete[] tr->pp_hst[i];
    CC(cudaFree(tr->pp));
}

void extents(const Particle *pp, const int nv, const int nm, /**/ TicketS *t) {
    t->llo->resize(nm); t->hhi->resize(nm);
    if (nm) minmax(pp, nv, nm, /**/ t->llo->DP, t->hhi->DP);
    dSync();
}

int pack(const Particle *pp, const int nv, const int nm, /**/ TicketS *t) {
    return sub::pack(t->llo->D, t->hhi->D, pp, nv, nm, /**/ t->pp_hst, t->counts);
}

void post_recv(/**/ TicketCom *tc, TicketR *tr) {
    sub::post_recv(tc->cart, tc->dstranks, tc->recv_tags, tc->btc, tc->btp, /**/ tr->counts, tr->pp_hst, &tc->rreq);
}

void post_send(int nv, const TicketS *ts, /**/ TicketCom *tc) {
    if (!tc->first) sub::wait_req(&tc->sreq);
    sub::post_send(tc->cart, tc->dstranks, tc->btc, tc->btp, nv, ts->counts, ts->pp_hst, /**/ &tc->sreq);
}

void wait_recv(TicketCom *t) {
    sub::wait_req(&t->rreq);
}

int unpack(int nv, int nbulk, /**/ TicketR *tr) {
    tr->counts[0] = nbulk;
    return sub::unpack(tr->counts, tr->pp_hst, nv, /**/ tr->pp);
}

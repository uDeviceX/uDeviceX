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

void alloc_ticketD(/*io*/ basetags::TagGen *tg, /**/ TicketD *t) {
    l::m::Comm_dup(m::cart, &t->cart);
    sub::ini_comm(t->cart, /**/ t->rank, t->r.tags);
    sub::ini_S(/**/ &t->s);
    sub::ini_R(&t->s, /**/ &t->r);
    t->first = true;
    mpDeviceMalloc(&t->subi_lo);
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketD(/**/ TicketD *t) {
    sub::fin_S(/**/ &t->s);
    sub::fin_R(/**/ &t->r);
    CC(cudaFree(t->subi_lo));
}

void alloc_ticketI(/*io*/ basetags::TagGen *tg, /**/ TicketI *t) {
    t->first = true;
    sub::ini_SRI(/**/ &t->sii, &t->rii);
    t->bt = get_tag(tg);
}

void free_ticketI(/**/ TicketI *t) {
    sub::fin_SRI(/**/ &t->sii, &t->rii);
}

void alloc_ticketU(TicketU *t) {
    mpDeviceMalloc(&t->subi_re);
    mpDeviceMalloc(&t->iidx);
    mpDeviceMalloc(&t->pp_re);
}

void free_ticketU(TicketU *t) {
    CC(cudaFree(t->subi_re));
    CC(cudaFree(t->iidx));
    CC(cudaFree(t->pp_re));
}

void alloc_ticketUI(TicketUI *t) {
    mpDeviceMalloc(&t->ii_re); 
}

void free_ticketUI(TicketUI *t) {
    CC(cudaFree(t->ii_re));
}

void alloc_work(Work *w) {
    scan::alloc_work(XS*YS*ZS, /**/ &w->s);
}

void free_work(Work *w) {
    scan::free_work(/**/ &w->s);
}

void post_recv_pp(TicketD *t) {
    sub::post_recv(t->cart, t->rank, t->btc, t->btp, /**/ t->recv_sz_req, t->recv_pp_req, &t->r);
}
void post_recv_ii(const TicketD *td, TicketI *ti) {
    sub::post_recv_ii(td->cart, td->rank, td->r.tags, ti->bt, /**/ ti->recv_ii_req, &ti->rii);
}

void pack_pp(const flu::Quants *q, TicketD *t) {
    if (q->n) {
        sub::halo(q->pp, q->n, /**/ &t->s);
        sub::scan(q->n, /**/ &t->s);
        sub::pack_pp(q->pp, q->n, /**/ &t->s);
        dSync();
    }
}    

void pack_ii(const int n, const flu::QuantsI *q, const TicketD *td, TicketI *ti) {
    if (n) sub::pack_ii(q->ii, n, &td->s, /**/ &ti->sii);
    dSync();
}

void send_pp(TicketD *t) {
    if (!t->first) {
        sub::waitall(t->send_sz_req);
        sub::waitall(t->send_pp_req);
    }
    t->first = false;
    t->nbulk = sub::send_sz(t->cart, t->rank, t->btc, /**/ &t->s, t->send_sz_req);
    sub::send_pp(t->cart, t->rank, t->btp, /**/ &t->s, t->send_pp_req);
}

void send_ii(const TicketD *td, TicketI *ti) {
    if (!ti->first) sub::waitall(ti->send_ii_req);
    ti->first = false;
    sub::send_ii(td->cart, td->rank, td->s.size, ti->bt, /**/ &ti->sii, ti->send_ii_req);
}

void bulk(flu::Quants *q, TicketD *t) {
    int n = q->n, *count = q->cells->count;
    CC(cudaMemsetAsync(count, 0, sizeof(int)*XS*YS*ZS));
    if (n)
    k_common::subindex_local<false><<<k_cnf(n)>>>(n, (float2*)q->pp, /*io*/ count, /*o*/ t->subi_lo);
}

void recv_pp(TicketD *t) {
    sub::waitall(t->recv_sz_req);
    sub::recv_count(/**/ &t->r, &t->nhalo);
    sub::waitall(t->recv_pp_req);
}

void recv_ii(TicketI *t) {
    sub::waitall(t->recv_ii_req);
}

void unpack_pp(const TicketD *td, /**/ flu::Quants *q, TicketU *tu, /*w*/ Work *w) {
    const int nhalo = td->nhalo;
    
    int *start = q->cells->start;
    int *count = q->cells->count;
    
    if (nhalo) {
        sub::unpack_pp(nhalo, &td->r, /**/ tu->pp_re);
        sub::subindex_remote(nhalo, &td->r, /*io*/ tu->pp_re, count, /**/ tu->subi_re);
    }

    scan::scan(count, XS*YS*ZS, /**/ start, /*w*/ &w->s);
    
    // k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>(XS*YS*ZS, (int4*)count, /**/ (uchar4*)w->count_zip);
    // l::scan::d::scan(w->count_zip, XS*YS*ZS, /**/ (uint*)start);
}

void unpack_ii(const TicketD *td, const TicketI *ti, TicketUI *tui) {
    const int nhalo = td->nhalo;
    if (nhalo) sub::unpack_ii(nhalo, &td->r, &ti->rii, /**/ tui->ii_re);    
}

void gather_pp(const TicketD *td, /**/ flu::Quants *q, TicketU *tu, flu::TicketZ *tz) {
    const int nhalo = td->nhalo, nbulk = td->nbulk;
    
    int n = q->n;
    int *start = q->cells->start;
    
    Particle *pp = q->pp, *pp0 = q->pp0;

    if (n)
        sub::dev::scatter<<<k_cnf(n)>>>(false, td->subi_lo,  n, start, /**/ tu->iidx);

    if (nhalo)
        sub::dev::scatter<<<k_cnf(nhalo)>>>(true, tu->subi_re, nhalo, start, /**/ tu->iidx);
    n = nbulk + nhalo;
    
    if (n)
    sub::dev::gather_pp<<<k_cnf(n)>>>((float2*)pp, (float2*)tu->pp_re, n, tu->iidx,
                                          /**/ (float2*)pp0, tz->zip0, tz->zip1);

    q->n = n;

    /* swap */
    q->pp = pp0; q->pp0 = pp; 
}

void gather_ii(const int n, const TicketU *tu, const TicketUI *tui , /**/ flu::QuantsI *q) {
    int *ii = q->ii, *ii0 = q->ii0;

    if (n) sub::dev::gather_id<<<k_cnf(n)>>>(ii, tui->ii_re, n, tu->iidx, /**/ ii0);

    /* swap */
    q->ii = ii0; q->ii0 = ii;
}

struct TicketD { /* distribution */
    MPI_Comm cart;
    int rank[27];
    MPI_Request send_sz_req[27], recv_sz_req[27];
    MPI_Request send_pp_req[27], recv_pp_req[27];
    MPI_Request send_ii_req[27], recv_ii_req[27];
    bool first = true;
    sub::Distr distr;
    uchar4 *subi_lo;           /* local subindices */
    int nhalo, nbulk;
};

struct Work {
    uchar4 *subi_re;           /* remote subindices */
    uint   *iidx;              /* scatter indices   */
    Particle *pp_re;           /* remote particles  */
    int *ii_re;                /* remote ids        */
    unsigned char *count_zip;
};

void alloc_ticketD(TicketD *t) {
    l::m::Comm_dup(m::cart, &t->cart);
    sub::ini_comm(t->cart, /**/ t->rank, t->distr.r.tags);
    sub::ini_S(/**/ &t->distr.s);
    sub::ini_R(&t->distr.s, /**/ &t->distr.r);
    t->first = true;
    mpDeviceMalloc(&t->subi_lo);
}

void free_ticketD(/**/ TicketD *t) {
    sub::fin_S(/**/ &t->distr.s);
    sub::fin_R(/**/ &t->distr.r);
    CC(cudaFree(t->subi_lo));
}

void alloc_work(Work *w) {
    mpDeviceMalloc(&w->subi_re);
    mpDeviceMalloc(&w->iidx);
    mpDeviceMalloc(&w->pp_re);
    if (global_ids) mpDeviceMalloc(&w->ii_re);
    CC(cudaMalloc(&w->count_zip, sizeof(w->count_zip[0])*XS*YS*ZS));
}

void free_work(Work *w) {
    CC(cudaFree(w->subi_re));
    CC(cudaFree(w->iidx));
    CC(cudaFree(w->pp_re));
    if (global_ids) CC(cudaFree(w->ii_re));
    CC(cudaFree(w->count_zip));
}

void post_recv(TicketD *t) {
    sub::Distr *D = &t->distr;
    D->post_recv(t->cart, t->rank, /**/ t->recv_sz_req, t->recv_pp_req);
    if (global_ids) D->post_recv_ii(t->cart, t->rank, /**/ t->recv_ii_req);        
}

void pack(flu::Quants *q, TicketD *t) {
    sub::Distr *D = &t->distr;
    if (q->n) {
        D->halo(q->pp, q->n);
        D->scan(q->n);
        D->pack_pp(q->pp, q->n);
        if (global_ids) D->pack_ii(q->ii, q->n);
        dSync();
    }
}    

void send(TicketD *t) {
    sub::Distr *D = &t->distr;
    if (!t->first) {
        D->waitall(t->send_sz_req);
        D->waitall(t->send_pp_req);
        if (global_ids) D->waitall(t->send_ii_req);
    }
    t->first = false;
    t->nbulk = D->send_sz(t->cart, t->rank, t->send_sz_req);
    D->send_pp(t->cart, t->rank, t->send_pp_req);
    if (global_ids) D->send_ii(t->cart, t->rank, t->send_ii_req);
}

void bulk(flu::Quants *q, TicketD *t) {
    int n = q->n, *count = q->cells->count;
    CC(cudaMemsetAsync(count, 0, sizeof(int)*XS*YS*ZS));
    if (n)
    k_common::subindex_local<false><<<k_cnf(n)>>>(n, (float2*)q->pp, /*io*/ count, /*o*/ t->subi_lo);
}

void recv(TicketD *t) {
    sub::Distr *D = &t->distr;
    D->waitall(t->recv_sz_req);
    D->recv_count(&t->nhalo);
    D->waitall(t->recv_pp_req);
    if (global_ids) D->waitall(t->recv_ii_req);
}

void unpack(flu::Quants *q, TicketD *td, flu::TicketZ *tz, Work *w) {
    const int nhalo = td->nhalo, nbulk = td->nbulk;
    sub::Distr *D = &td->distr;

    int n = q->n;
    int *start = q->cells->start;
    int *count = q->cells->count;

    Particle *pp = q->pp, *pp0 = q->pp0;
    int *ii = q->ii, *ii0 = q->ii0;
    
    if (nhalo) {
        D->unpack_pp(nhalo, /*o*/ w->pp_re);
        if (global_ids) D->unpack_ii(nhalo, w->ii_re);
        D->subindex_remote(nhalo, /*io*/ w->pp_re, count, /**/ w->subi_re);
    }
    
    k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>(XS*YS*ZS, (int4*)count, /**/ (uchar4*)w->count_zip);
    l::scan::d::scan(w->count_zip, XS*YS*ZS, /**/ (uint*)start);

    if (n)
        sub::dev::scatter<<<k_cnf(n)>>>(false, td->subi_lo,  n, start, /**/ w->iidx);

    if (nhalo)
        sub::dev::scatter<<<k_cnf(nhalo)>>>(true, w->subi_re, nhalo, start, /**/ w->iidx);
    n = nbulk + nhalo;
    if (n) {
        sub::dev::gather_pp<<<k_cnf(n)>>>((float2*)pp, (float2*)w->pp_re, n, w->iidx,
                                          /**/ (float2*)pp0, tz->zip0, tz->zip1);
        if (global_ids) sub::dev::gather_id<<<k_cnf(n)>>>(ii, w->ii_re, n, w->iidx, /**/ ii0);
    }

    q->n = n;

    /* swap */
    q->pp = pp0; q->pp0 = pp; 
    if (global_ids) {
        q->ii = ii0; q->ii0 = ii;
    }    
}

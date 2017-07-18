struct TicketD { /* distribution */
    MPI_Comm cart;
    int rank[27];
    MPI_Request send_sz_req[27], recv_sz_req[27];
    MPI_Request send_pp_req[27], recv_pp_req[27];
    MPI_Request send_ii_req[27], recv_ii_req[27];
    bool first = true;
    sub::Send s;
    sub::Recv r;
    uchar4 *subi_lo;           /* local subindices */
    int nhalo, nbulk;
};

struct TicketU { /* unpack ticket */
    uchar4 *subi_re;           /* remote subindices */
    Particle *pp_re;           /* remote particles  */
    int *ii_re;                /* remote ids        */
};

struct Work {
    // TODO rm these
    uchar4 *subi_re;           /* remote subindices */
    Particle *pp_re;           /* remote particles  */
    int *ii_re;                /* remote ids        */

    uint*iidx;              /* scatter indices   */
    unsigned char *count_zip;
};

void alloc_ticketD(TicketD *t) {
    l::m::Comm_dup(m::cart, &t->cart);
    sub::ini_comm(t->cart, /**/ t->rank, t->r.tags);
    sub::ini_S(/**/ &t->s);
    sub::ini_R(&t->s, /**/ &t->r);
    t->first = true;
    mpDeviceMalloc(&t->subi_lo);
}

void free_ticketD(/**/ TicketD *t) {
    sub::fin_S(/**/ &t->s);
    sub::fin_R(/**/ &t->r);
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
    sub::post_recv(t->cart, t->rank, /**/ t->recv_sz_req, t->recv_pp_req, &t->r);
    if (global_ids) sub::post_recv_ii(t->cart, t->rank, /**/ t->recv_ii_req, &t->r);
}

void pack(flu::Quants *q, TicketD *t) {
    if (q->n) {
        sub::halo(q->pp, q->n, /**/ &t->s);
        sub::scan(q->n, /**/ &t->s);
        sub::pack_pp(q->pp, q->n, /**/ &t->s);
        if (global_ids) sub::pack_ii(q->ii, q->n, /**/ &t->s);
        dSync();
    }
}    

void send(TicketD *t) {
    if (!t->first) {
        sub::waitall(t->send_sz_req);
        sub::waitall(t->send_pp_req);
        if (global_ids) sub::waitall(t->send_ii_req);
    }
    t->first = false;
    t->nbulk = sub::send_sz(t->cart, t->rank, /**/ &t->s, t->send_sz_req);
    sub::send_pp(t->cart, t->rank, /**/ &t->s, t->send_pp_req);
    if (global_ids) sub::send_ii(t->cart, t->rank, /**/ &t->s, t->send_ii_req);
}

void bulk(flu::Quants *q, TicketD *t) {
    int n = q->n, *count = q->cells->count;
    CC(cudaMemsetAsync(count, 0, sizeof(int)*XS*YS*ZS));
    if (n)
    k_common::subindex_local<false><<<k_cnf(n)>>>(n, (float2*)q->pp, /*io*/ count, /*o*/ t->subi_lo);
}

void recv(TicketD *t) {
    sub::waitall(t->recv_sz_req);
    sub::recv_count(/**/ &t->r, &t->nhalo);
    sub::waitall(t->recv_pp_req);
    if (global_ids) sub::waitall(t->recv_ii_req);
}

void unpack(flu::Quants *q, TicketD *td, flu::TicketZ *tz, Work *w) {
    const int nhalo = td->nhalo, nbulk = td->nbulk;
    
    int n = q->n;
    int *start = q->cells->start;
    int *count = q->cells->count;

    Particle *pp = q->pp, *pp0 = q->pp0;
    int      *ii = q->ii, *ii0 = q->ii0;
    
    if (nhalo) {
        sub::unpack_pp(nhalo, /**/ &td->r, w->pp_re);
        if (global_ids) sub::unpack_ii(nhalo, /**/ &td->r, w->ii_re);
        sub::subindex_remote(nhalo, &td->r, /*io*/ w->pp_re, count, /**/ w->subi_re);
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

void unpack_pp(flu::Quants *q, TicketD *td, TicketU *tu, Work *w) {
    const int nhalo = td->nhalo;
    
    int *start = q->cells->start;
    int *count = q->cells->count;
    
    if (nhalo) {
        sub::unpack_pp(nhalo, /**/ &td->r, w->pp_re);
        sub::subindex_remote(nhalo, &td->r, /*io*/ w->pp_re, count, /**/ w->subi_re);
    }
    
    k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>(XS*YS*ZS, (int4*)count, /**/ (uchar4*)w->count_zip);
    l::scan::d::scan(w->count_zip, XS*YS*ZS, /**/ (uint*)start);
}

void gather_pp(flu::Quants *q, TicketD *td, TicketU *tu, flu::TicketZ *tz, Work *w) {
    const int nhalo = td->nhalo, nbulk = td->nbulk;
    
    int n = q->n;
    int *start = q->cells->start;
    
    Particle *pp = q->pp, *pp0 = q->pp0;

    if (n)
        sub::dev::scatter<<<k_cnf(n)>>>(false, td->subi_lo,  n, start, /**/ w->iidx);

    if (nhalo)
        sub::dev::scatter<<<k_cnf(nhalo)>>>(true, w->subi_re, nhalo, start, /**/ w->iidx);
    n = nbulk + nhalo;
    
    if (n)
    sub::dev::gather_pp<<<k_cnf(n)>>>((float2*)pp, (float2*)w->pp_re, n, w->iidx,
                                          /**/ (float2*)pp0, tz->zip0, tz->zip1);

    q->n = n;

    /* swap */
    q->pp = pp0; q->pp0 = pp; 
}

void unpack_ii(flu::Quants *q, TicketD *td, flu::TicketZ *tz, Work *w) {
    const int nhalo = td->nhalo, nbulk = td->nbulk;
    
    int n = q->n;
    int *start = q->cells->start;
    int *count = q->cells->count;

    Particle *pp = q->pp, *pp0 = q->pp0;
    int      *ii = q->ii, *ii0 = q->ii0;
    
    if (nhalo) {
        sub::unpack_pp(nhalo, /**/ &td->r, w->pp_re);
        if (global_ids) sub::unpack_ii(nhalo, /**/ &td->r, w->ii_re);
        sub::subindex_remote(nhalo, &td->r, /*io*/ w->pp_re, count, /**/ w->subi_re);
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

struct TicketD { /* distribution */
    MPI_Comm cart;
    int rank[27];
    MPI_Request send_size_req[27], recv_size_req[27];
    MPI_Request send_mesg_req[27], recv_mesg_req[27];
    bool first = true;
    sub::Distr distr; /* was odstr; */
};

struct Work {
    uchar4 *subi_lo, *subi_re; /* local remote subindices */
    uint   *iidx;              /* scatter indices */
    Particle *pp_re;           /* remote particles */
    unsigned char *count_zip;
};

void alloc_ticketD(TicketD *t) {
    l::m::Comm_dup(m::cart, &t->cart);
    t->distr.ini(t->cart, t->rank);
    t->first = true;
}

void free_ticketD(/**/ TicketD *t) {
    t->distr.fin();
}

void alloc_work(Work *w) {
    mpDeviceMalloc(&w->subi_lo);
    mpDeviceMalloc(&w->subi_re);
    mpDeviceMalloc(&w->iidx);
    mpDeviceMalloc(&w->pp_re);
    CC(cudaMalloc(&w->count_zip, sizeof(w->count_zip[0])*XS*YS*ZS));
}

void free_work(Work *w) {
    CC(cudaFree(w->subi_lo));
    CC(cudaFree(w->subi_re));
    CC(cudaFree(w->iidx));
    CC(cudaFree(w->pp_re));
    CC(cudaFree(w->count_zip));
}

void distr(flu::Quants *q, TicketD *td, flu::TicketZ *tz, Work *w) {
    MPI_Comm cart = td->cart; /* can be a copy */
    int *rank = td->rank; /* arrays */
    int *send_size_req = td->send_size_req;
    int *recv_size_req = td->recv_size_req;
    MPI_Request *send_mesg_req = td->send_mesg_req;
    MPI_Request *recv_mesg_req = td->recv_mesg_req;
    bool *qfirst = &td->first; /* shoud be updated */

    float4  *zip0 = tz->zip0;
    ushort4 *zip1 = tz->zip1;

    uchar4 *subi_lo = w->subi_lo; /* arrays */
    uchar4 *subi_re = w->subi_re;
    uint   *iidx = w->iidx;
    Particle *pp_re = w->pp_re;
    unsigned char *count_zip = w->count_zip;
    Particle *pp0 = q->pp0;

    int n = q->n;
    bool first = *qfirst;
    Particle *pp = q->pp;
  
    int nbulk, nhalo;
    td->distr.post_recv(cart, rank, /**/ recv_size_req, recv_mesg_req);
    if (n) {
        td->distr.halo(pp, n);
        td->distr.scan(n);
        td->distr.pack_pp(pp, n);
        dSync();
    }
    if (!first) {
        td->distr.waitall(send_size_req);
        td->distr.waitall(send_mesg_req);
    }
    first = false;
    nbulk = td->distr.send_sz(cart, rank, send_size_req);
    td->distr.send_msg(cart, rank, send_mesg_req);

    CC(cudaMemsetAsync(q->cells->count, 0, sizeof(int)*XS*YS*ZS));
    if (n)
    k_common::subindex_local<false><<<k_cnf(n)>>>
        (n, (float2*)pp, /*io*/ q->cells->count, /*o*/ subi_lo);

    td->distr.waitall(recv_size_req);
    td->distr.recv_count(&nhalo);
    td->distr.waitall(recv_mesg_req);

    if (nhalo)
    td->distr.unpack(nhalo, /*io*/ q->cells->count, /*o*/ subi_re, pp_re);

    k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>
        (XS*YS*ZS, (int4*)q->cells->count, /**/ (uchar4*)count_zip);
    l::scan::d::scan(count_zip, XS*YS*ZS, /**/ (uint*)q->cells->start);

    if (n)
    sub::dev::scatter<<<k_cnf(n)>>>
        (false, subi_lo,  n, q->cells->start, /**/ iidx);

    if (nhalo)
    sub::dev::scatter<<<k_cnf(nhalo)>>>
        (true, subi_re, nhalo, q->cells->start, /**/ iidx);

    n = nbulk + nhalo;
    if (n)
    sub::dev::gather_pp<<<k_cnf(n)>>>((float2*)pp, (float2*)pp_re, n, iidx,
                                      /**/ (float2*)pp0, zip0, zip1);

    q->n = n;
    *qfirst = first;
    q->pp = pp0; q->pp0 = pp; /* swap */
}

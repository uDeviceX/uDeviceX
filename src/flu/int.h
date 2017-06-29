struct Quants {
    Particle *pp, *pp0;
    int      *ii, *ii0;
    int       n;
    Clist *cells;
    Particle *pp_hst;
}; 

struct TicketZ { /* zip */
    float4  *zip0;
    ushort4 *zip1;
};

void alloc_quants(Quants *q) {
    q->n = 0;
    mpDeviceMalloc(&q->pp); mpDeviceMalloc(&q->pp0);
    mpDeviceMalloc(&q->ii); mpDeviceMalloc(&q->ii0);
    q->cells = new Clist(XS, YS, ZS);
    q->pp_hst = new Particle[MAX_PART_NUM];
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp)); CC(cudaFree(q->pp0));
    CC(cudaFree(q->ii)); CC(cudaFree(q->ii0));
    delete q->cells;
    delete[] q->pp_hst;
}

void alloc_ticketZ(/**/ TicketZ *t) {
    mpDeviceMalloc(&t->zip0);
    mpDeviceMalloc(&t->zip1);
}

void free_ticketZ(/**/ TicketZ *t) {
    float4  *zip0 = t->zip0;
    ushort4 *zip1 = t->zip1;
    cudaFree(zip0);
    cudaFree(zip1);
}

void get_ticketZ(Quants q, /**/ TicketZ *t) {
    if (q.n == 0) return;
    float4  *zip0 = t->zip0;
    ushort4 *zip1 = t->zip1;
    assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: implicit dependency */
    sub::dev::zip<<<(q.n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(zip0, zip1, (float*)q.pp, q.n);
}

void gen_quants(Quants *q) {
    q->n = sub::gen(q->pp, q->pp_hst);
}

void strt_quants(const int id, Quants *q) {
    q->n = sub::strt(id, q->pp, /* w */ q->pp_hst);
}

void strt_dump(const int id, const Quants q) {
    sub::strt_dump(id, q.n, q.pp, /* w */ q.pp_hst);
}

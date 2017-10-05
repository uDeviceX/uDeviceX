static void unpack_bulk_ii(int nc, const Pack *p, /**/ int *ii) {
    void *src = p->hii.data[frag_bulk];
    memcpy(ii, src, nc * sizeof(int));
}

void unpack_bulk(const Pack *p, /**/ rbc::Quants *q) {
    int nc, nv, n;
    nc = p->hpp.counts[frag_bulk];
    nv = q->nv;
    n = nc * nv;
    
    if (n) CC(d::MemcpyAsync(q->pp, p->dpp.data[frag_bulk], n * sizeof(Particle), D2D));
    q->nc = nc;
    q->n = n;

    if (rbc_ids)
        unpack_bulk_ii(nc, p, /**/ q->ii);
}

static void unpack_halo_ii(const hBags *hii, /**/ int *ii) {
    void *src;
    int i, s, c;

    s = 0;
    for (i = 0; i < NFRAGS; ++i) {
        c   = hii->counts[i];
        src = hii->data[i];        
        memcpy(ii + s, src, c * sizeof(int));
        s += c;
    }
}

void unpack_halo(const Unpack *u, /**/ rbc::Quants *q) {
    int nc, nv, n, i, s, nctot;
    size_t size;
    nv = q->nv;
    s = q->n;
    nctot = q->nc;
    
    for (i = 0; i < NFRAGS; ++i) {
        nc = u->hpp.counts[i];
        n  = nc * nv; 
        size = n * sizeof(Particle);
        if (nc) {
            CC(d::MemcpyAsync(q->pp + s, u->hpp.data[i], size, H2D));
            KL(dev::shift_one_frag, (k_cnf(n)), (n, i, /**/ q->pp + s));
        }
        s += n;
        nctot += nc;
    }

    if (rbc_ids)
        unpack_halo_ii(&u->hii, /**/ q->ii + q->nc);

    q->n = s;
    q->nc = nctot;
}

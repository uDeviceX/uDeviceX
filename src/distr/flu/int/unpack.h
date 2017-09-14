void unpack_pp(/**/ Unpack *u) {
    int nhalo;
    nhalo = unpack_pp(u->hpp, /**/ u->ppre);
    u->nhalo = nhalo;
}

void unpack_ii(/**/ Unpack *u) {
    unpack_ii(u->hii, /**/ u->iire);
}

void unpack_cc(/**/ Unpack *u) {
    unpack_ii(u->hcc, /**/ u->ccre);
}

void bulk(/**/ Quants *q) {
    ini_counts(&q->cells);
    subindex_local(q->n, q->pp, /**/ &q->cells, &q->tcells);
}

void halo(const Unpack *u, /**/ Quants *q) {
    subindex_remote(u->nhalo, u->ppre, /**/ &q->cells, &q->tcells);
}

void gather(const Pack *p, const Unpack *u, /**/ Quants *q) {
    int nold, n, nhalo, nbulk;
    Particle *pp, *pp0;
    nold = q->n;
    nhalo = u->nhalo;
    nbulk = p->nbulk;
    n = nbulk + nhalo;
    pp = q->pp;
    pp0 = q->pp0;
    
    build_map(nbulk, nhalo, /**/ &q->cells, &q->tcells);

    gather_pp(pp, u->ppre, &q->tcells, n, /**/ pp0);

    // TODO
    // void gather_ii(const int *iilo, const int *iire, const Ticket *t, int nout, /**/ int *iiout);

    q->n = n;
    q->pp = pp0;
    q->pp0 = p;
}

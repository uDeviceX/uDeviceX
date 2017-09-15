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

void gather(const Pack *p, const Unpack *u, /**/ Quants *q, QuantsI *qi, QuantsI *qc) {
    int n, nold, nhalo, nbulk;
    Particle *pp, *pp0;
    nold = q->n;
    nhalo = u->nhalo;
    nbulk = p->nbulk;
    n = nbulk + nhalo;
    pp = q->pp; pp0 = q->pp0;    
    
    build_map(nold, nhalo, /**/ &q->cells, &q->tcells);

    gather_pp(pp, u->ppre, &q->tcells, n, /**/ pp0);

    int *ii, *ii0, *cc, *cc0;
    ii = qi->ii; ii0 = qi->ii0;
    cc = qc->ii; cc0 = qc->ii0;
    
    if (global_ids)    gather_ii(ii, u->iire, &q->tcells, n, /**/ ii0);
    if (multi_solvent) gather_ii(cc, u->ccre, &q->tcells, n, /**/ cc0);

    q->n = n;

    /* swap pointers */

    q->pp = pp0;
    q->pp0 = pp;

    if (global_ids) {
        qi->ii = ii0;
        qi->ii0 = ii;
    }

    if (multi_solvent) {
        qc->ii = cc0;
        qc->ii0 = cc;
    }
}

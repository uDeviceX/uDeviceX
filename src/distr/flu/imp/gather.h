void bulk(/**/ Quants *q) {
    PartList lp;
    lp.pp = q->pp;
    lp.deathlist = NULL;
    ini_counts(&q->cells);
    subindex_local(q->n, lp, /**/ &q->cells, &q->mcells);
}

void halo(const Unpack *u, /**/ Quants *q) {
    PartList lp;
    lp.pp = u->ppre;
    lp.deathlist = NULL;
    subindex_remote(u->nhalo, lp, /**/ &q->cells, &q->mcells);
}

void gather(const Pack *p, const Unpack *u, /**/ Quants *q) {
    int n, nold, nhalo, nbulk;
    Particle *pp, *pp0;
    nold = q->n;
    nhalo = u->nhalo;
    nbulk = p->nbulk;
    n = nbulk + nhalo;
    pp = q->pp; pp0 = q->pp0;    

    const int nn[] = {nold, nhalo};
    build_map(nn, /**/ &q->cells, &q->mcells);

    gather_pp(pp, u->ppre, &q->mcells, n, /**/ pp0);

    int *ii, *ii0, *cc, *cc0;
    ii = q->ii; ii0 = q->ii0;
    cc = q->cc; cc0 = q->cc0;
    
    if (global_ids)    gather_ii(ii, u->iire, &q->mcells, n, /**/ ii0);
    if (multi_solvent) gather_ii(cc, u->ccre, &q->mcells, n, /**/ cc0);

    q->n = n;

    /* swap pointers */

    q->pp = pp0;
    q->pp0 = pp;

    if (global_ids) {
        q->ii = ii0;
        q->ii0 = ii;
    }

    if (multi_solvent) {
        q->cc = cc0;
        q->cc0 = cc;
    }
}

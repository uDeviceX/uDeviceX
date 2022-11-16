void dflu_bulk(PartList lp, /**/ FluQuants *q) {
    UC(clist_ini_counts(q->cells));
    UC(clist_subindex_local(q->n, lp, /**/ q->cells, q->mcells));
}

void dflu_halo(const DFluUnpack *u, /**/ FluQuants *q) {
    PartList lp;
    lp.pp = u->ppre;
    lp.deathlist = NULL;
    UC(clist_subindex_remote(u->nhalo, lp, /**/ q->cells, q->mcells));
}

void dflu_gather(int ndead, const DFluPack *p, const DFluUnpack *u, /**/ FluQuants *q) {
    int n, nold, nhalo, nbulk;
    Particle *pp, *pp0;
    nold = q->n;
    nhalo = u->nhalo;
    nbulk = nold - p->nhalo - ndead;
    n = nbulk + nhalo;
    pp = q->pp; pp0 = q->pp0;    

    const int nn[] = {nold, nhalo};
    UC(clist_build_map(nn, /**/ q->cells, q->mcells));

    UC(clist_gather_pp(pp, u->ppre, q->mcells, n, /**/ pp0));

    int *ii, *ii0, *cc, *cc0;
    ii = q->ii; ii0 = q->ii0;
    cc = q->cc; cc0 = q->cc0;
    
    if (p->hii) UC(clist_gather_ii(ii, u->iire, q->mcells, n, /**/ ii0));
    if (p->hcc) UC(clist_gather_ii(cc, u->ccre, q->mcells, n, /**/ cc0));

    q->n = n;

    /* swap pointers */

    q->pp = pp0;
    q->pp0 = pp;

    if (p->hii) {
        q->ii = ii0;
        q->ii0 = ii;
    }

    if (p->hcc) {
        q->cc = cc0;
        q->cc0 = cc;
    }
}

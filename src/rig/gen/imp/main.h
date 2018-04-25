static void gen_ids(MPI_Comm comm, int n, int *ii) {
    int i, i0 = 0, count = 1;
    MC(m::Exscan(&n, &i0, count, MPI_INT, MPI_SUM, comm));
    for (i = 0; i < n; ++i) ii[i] = i + i0;
}

void rig_gen_from_solvent(const Coords *coords, MPI_Comm cart, RigGenInfo rgi, /*io*/ FluInfo fi, /**/ RigInfo ri) {
    int3 L;
    int *ids, nflu, *ll_hst, *ll_dev;
    Particle *pp_flu_hst;
    bool hasid0;

    L = subdomain(coords);
    nflu = *fi.n;

    EMALLOC(ri.ns, &ids);
    EMALLOC(nflu, &pp_flu_hst);
    EMALLOC(nflu, &ll_hst);
    Dalloc(&ll_dev, nflu);
    cD2H(pp_flu_hst, fi.pp, nflu);

    UC(gen_ids(cart, ri.ns, ids));

    hasid0 = (ri.ns && ids[0] == 0);     
    
    UC(extract_template(L, cart, rgi, nflu, fi.pp, pp_flu_hst, ri.ns, hasid0, ri.ss,
                        /**/ ri.nps, ri.rr0, /*w*/ ll_dev, ll_hst));

    UC(kill_solvent(rgi, MAX_SOLIDS, L, cart, ri.ns,
                    /*io*/ fi.n, pp_flu_hst, fi.pp,
                    /*w*/ ll_dev, ll_hst));

    // TODO props
    
    Dfree(ll_dev);
    EFREE(pp_flu_hst);
    EFREE(ll_hst);
    EFREE(ids);
}

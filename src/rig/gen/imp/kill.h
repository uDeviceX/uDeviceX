static void remove_hst(const int *labels, int *n, Particle *pp) {
    int i, j;
    for (i = j = 0; i < *n; ++i) {
        if (labels[i] == IN)
            pp[j++] = pp[i];
    }
    *n = j;
}

static void kill_solvent(RigGenInfo rgi, int maxm, int3 L, MPI_Comm cart, int nm,
                         /*io*/ int *n, Particle *pp_hst, Particle *pp_dev,
                         /*w*/ int *ll_dev, int *ll_hst) {
    int nmall, pdir;
    Particle *i_ppall;
    nmall = nm;
    pdir = rig_pininfo_get_pdir(rgi.pi);

    Dalloc(&i_ppall, maxm * rgi.nv);
    cD2D(i_ppall, rgi.pp, nm * rgi.nv);
    
    UC(exchange_mesh(maxm, L, cart, rgi.nv, /*io*/ &nmall, i_ppall, NULL));
    UC(compute_labels(pdir, *n, pp_dev, rgi.nt, rgi.nv, nmall, rgi.tt, i_ppall, IN, OUT, /**/ ll_dev));

    cD2H(ll_hst, ll_dev, *n);
    remove_hst(ll_hst, n, pp_hst);
    cH2D(pp_dev, pp_hst, *n);

    Dfree(i_ppall);
}

static void remove_hst(const int *labels, int *n, Particle *pp) {
    int i, j;
    for (i = j = 0; i < *n; ++i) {
        if (labels[i] == IN)
            pp[j++] = pp[i];
    }
    *n = j;
}

static void kill_solvent(int pdir, int maxm, int3 L, MPI_Comm cart, int nv, int nt, int nm, const int4 *tt,
                         const Particle *i_pp,
                         /*io*/ int *n, Particle *pp_hst, Particle *pp_dev,
                         /*w*/ int *ll_dev, int *ll_hst) {
    int nmall;
    Particle *i_ppall;
    nmall = nm;

    Dalloc(&i_ppall, maxm * nv);
    
    UC(exchange_mesh(maxm, L, cart, nv, /*io*/ &nmall, i_ppall, NULL));
    UC(compute_labels(pdir, *n, pp_dev, nt, nv, nmall, tt, i_ppall, IN, OUT, /**/ ll_dev));

    cD2H(ll_hst, ll_dev, *n);
    remove_hst(ll_hst, n, pp_hst);
    cH2D(pp_dev, pp_hst, *n);

    Dfree(i_ppall);
}

static void fsi_halo_one_type(const PairParams *params, Fsi *fsi, Pap26 PP, Fop26 FF, const int counts[26]) {
    enum {NFRAGS = 26};
    int i, n, s;
    int27 starts;
    SolventWrap *wo = fsi->wo;
    const PaArray *parray = &wo->pa;
    float rnd = rnd_get(fsi->rgen);

    starts.d[0] = 0;
    for (i = s = 0; i < NFRAGS; ++i) starts.d[i + 1] = (s += counts[i]);
    n = starts.d[NFRAGS];

    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaCArray_v pav;

        pair_get_view_dpd_mirrored(params, &pv);
        parray_get_view(parray, &pav);
    
        KL(fsi_dev::halo, (k_cnf(n)), (pv, fsi->L, wo->starts, starts, PP, FF, pav, n, wo->n, rnd, /**/ (float*)wo->ff));
    }
    else {
        PairDPD pv;
        PaArray_v pav;

        pair_get_view_dpd(params, &pv);
        parray_get_view(parray, &pav);
    
        KL(fsi_dev::halo, (k_cnf(n)), (pv, fsi->L, wo->starts, starts, PP, FF, pav, n, wo->n, rnd, /**/ (float*)wo->ff));
    }
}

void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts) {
    enum {NFRAGS = 26};
    Pap26 pp;
    Fop26 ff;
    int i, j, start[NFRAGS];
    const int *counts;

    memset(start, 0, sizeof(start));
    
    for (i = 0; i < nw; i++) {
        counts = all_counts + NFRAGS * i;
        for (j = 0; j < NFRAGS; ++j) {
            pp.d[j] = all_pp.d[j] + start[j];
            ff.d[j] = all_ff.d[j] + start[j];
            start[j] += counts[j];
        }
        fsi_halo_one_type(prms[i], fsi, pp, ff, counts);
    }
}

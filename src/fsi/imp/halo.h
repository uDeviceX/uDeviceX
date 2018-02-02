template <typename Par>
static void halo_interactions(Par params, Fsi *fsi, Pap26 PP, Fop26 FF, int counts[26]) {
    int i, n, s;
    int27 starts;
    SolventWrap *wo = fsi->wo;
    const Cloud cloud = wo->c;
    float rnd = rnd_get(fsi->rgen);

    starts.d[0] = 0;
    for (i = s = 0; i < 26; ++i) starts.d[i + 1] = (s += counts[i]);
    n = starts.d[26];

    KL(dev::halo, (k_cnf(n)), (params, fsi->L, wo->starts, starts, PP, FF, cloud, n, wo->n, rnd, /**/ (float*)wo->ff));
}

void fsi_halo(const PairParams *params, Fsi *fsi, Pap26 PP, Fop26 FF, int counts[26]) {
    PairDPDCM pv;
    pair_get_view_dpd_mirrored(params, &pv);
    halo_interactions(pv, fsi, PP, FF, counts);
}

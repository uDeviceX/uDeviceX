template <typename Par>
static void bulk_one_wrap(Par params, PaWrap *pw, FoWrap *fw, Fsi *fsi) {
    int n0, n1;
    float rnd;
    const Particle *ppA = pw->pp;
    SolventWrap *wo = fsi->wo;
    Cloud cloud = wo->c;
    
    rnd = rnd_get(fsi->rgen);
    n0 = pw->n;
    n1 = wo->n;

    KL(dev::bulk, (k_cnf(3*n0)), (params, fsi->L, wo->starts, (float*)ppA, cloud, 
                                  n0, n1,
                                  rnd, (float*)fw->ff, (float*)wo->ff));
}

void fsi_bulk(const PairParams *params, Fsi *fsi, int nw, PaWrap *pw, FoWrap *fw) {
    if (nw == 0)
        return;

    PairDPDCM pv;
    pair_get_view_dpd_mirrored(params, &pv);
    
    for (int i = 0; i < nw; i++)
        bulk_one_wrap(pv, pw++, fw++, fsi);
}

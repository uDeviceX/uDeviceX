static void bulk_one_wrap(const PairParams *params, PaWrap *pw, FoWrap *fw, Fsi *fsi) {
    int n0, n1;
    float rnd;
    const Particle *ppA = pw->pp;
    SolventWrap *wo = fsi->wo;
    const PaArray *parray = &wo->pa;
    
    rnd = rnd_get(fsi->rgen);
    n0  = pw->n;
    n1  = wo->n;

    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaCArray_v paview;
        
        pair_get_view_dpd_mirrored(params, &pv);
        parray_get_view(parray, &paview);
        
        KL(fsi_dev::bulk, (k_cnf(3*n0)), (pv, fsi->L, wo->starts, (float*)ppA, paview, 
                                          n0, n1,
                                          rnd, (float*)fw->ff, (float*)wo->ff));
    }
    else {
        PairDPD pv;
        PaArray_v paview;

        pair_get_view_dpd(params, &pv);
        parray_get_view(parray, &paview);

        KL(fsi_dev::bulk, (k_cnf(3*n0)), (pv, fsi->L, wo->starts, (float*)ppA, paview, 
                                          n0, n1,
                                          rnd, (float*)fw->ff, (float*)wo->ff));
    }
}

void fsi_bulk(const PairParams *params, Fsi *fsi, int nw, PaWrap *pw, FoWrap *fw) {
    if (nw == 0)
        return;
    
    for (int i = 0; i < nw; i++)
        bulk_one_wrap(params, pw++, fw++, fsi);
}

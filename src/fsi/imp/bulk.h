template <typename Par>
static void bulk_one_wrap(Par params, PaWrap *pw, FoWrap *fw, Fsi *fsi) {
    int n0, n1;
    float rnd;
    const Particle *ppA = pw->pp;
    SolventWrap *wo = fsi->wo;
    const PaArray *parray = &wo->pa;
    
    rnd = rnd_get(fsi->rgen);
    n0  = pw->n;
    n1  = wo->n;

    if (parray_is_colored(parray)) {
        PaCArray_v paview;
        parray_get_view(parray, &paview);
        KL(dev::bulk, (k_cnf(3*n0)), (params, fsi->L, wo->starts, (float*)ppA, paview, 
                                      n0, n1,
                                      rnd, (float*)fw->ff, (float*)wo->ff));
    }
    else {
        PaArray_v paview;
        parray_get_view(parray, &paview);
        KL(dev::bulk, (k_cnf(3*n0)), (params, fsi->L, wo->starts, (float*)ppA, paview, 
                                      n0, n1,
                                      rnd, (float*)fw->ff, (float*)wo->ff));
    }
}

template <typename Par>
void bulk_interactions(Par params, Fsi *fsi, int nw, PaWrap *pw, FoWrap *fw) {
    if (nw == 0)
        return;
    
    for (int i = 0; i < nw; i++)
        bulk_one_wrap(params, pw++, fw++, fsi);
}

void fsi_bulk(const PairParams *params, Fsi *fsi, int nw, PaWrap *pw, FoWrap *fw) {
    if (multi_solvent) {
        PairDPDCM pv;
        pair_get_view_dpd_mirrored(params, &pv);
        bulk_interactions(pv, fsi, nw, pw, fw);
    }
    else {
        PairDPD pv;
        pair_get_view_dpd(params, &pv);
        bulk_interactions(pv, fsi, nw, pw, fw);
    }
}

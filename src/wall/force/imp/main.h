enum {
    TPP = 3 /* # threads per particle */
};

template<typename Par, typename Parray>
static void interactions(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff) {
    Coords_v coordsv;
    coords_get_view(c, &coordsv);
    
    KL(wf_dev::force,
       (k_cnf(TPP*n)),
       (params, wv, coordsv, parray, n, rnd_get(rnd), wa, /**/ (float*)ff));
}

void wall_force_apply(const PairParams *params, Wvel_v wv, const Coords *c, const PaArray *parray, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff) {
    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaArray_v pav;        
        UC(pair_get_view_dpd_mirrored(params, &pv));
        parray_get_view(parray, &pav);
        interactions(pv, wv, c, pav, n, rnd, wa, /**/ ff);
    }
    else {
        PairDPD pv;
        PaArray_v pav;
        UC(pair_get_view_dpd(params, &pv));
        parray_get_view(parray, &pav);
        interactions(pv, wv, c, pav, n, rnd, wa, /**/ ff);
    }
}

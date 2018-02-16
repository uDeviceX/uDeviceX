enum {
    TPP = 3 /* # threads per particle */
};

template <typename Par, typename Parray, typename Farray>
static void interactions(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ Farray farray) {
    Coords_v coordsv;
    coords_get_view(c, &coordsv);
    
    KL(wf_dev::force,
       (k_cnf(TPP*n)),
       (params, wv, coordsv, parray, n, rnd_get(rnd), wa, /**/ farray));
}

template <typename Par, typename Parray>
static void apply(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {
    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, wv, c, parray, n, rnd, wa, /**/ farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, wv, c, parray, n, rnd, wa, /**/ farray_v);        
    }
}

void wall_force_apply(const PairParams *params, Wvel_v wv, const Coords *c, const PaArray *parray, int n, RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {    
    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaArray_v pav;        
        UC(pair_get_view_dpd_mirrored(params, &pv));
        parray_get_view(parray, &pav);
        apply(pv, wv, c, pav, n, rnd, wa, /**/ farray);
    }
    else {
        PairDPD pv;
        PaArray_v pav;
        UC(pair_get_view_dpd(params, &pv));
        parray_get_view(parray, &pav);
        apply(pv, wv, c, pav, n, rnd, wa, /**/ farray);
    }
}

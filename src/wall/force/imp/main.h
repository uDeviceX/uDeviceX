enum {
    TPP = 3 /* # threads per particle */
};

template <typename Par, typename Wvel_v, typename Parray, typename Farray>
static void interactions(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ Farray farray) {
    Coords_v coordsv;
    coords_get_view(c, &coordsv);
    
    KL(wf_dev::force,
       (k_cnf(TPP*n)),
       (params, wv, coordsv, parray, n, rnd_get(rnd), wa, /**/ farray));
}

template <typename Par, typename Wvel_v, typename Parray>
static void stress_dispatch(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {
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

template <typename Par, typename Parray>
static void wvel_dispatch(Par params, const WvelStep *w, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {
    switch (wvel_get_type(w)) {
    case WALL_VEL_V_CSTE: {
        WvelCste_v wv;
        wvel_get_view(w, /**/ &wv);
        stress_dispatch(params, wv, c, parray, n, rnd, wa, /**/ farray);
        break;
    }
    case WALL_VEL_V_SHEAR: {
        WvelShear_v wv;
        wvel_get_view(w, /**/ &wv);
        stress_dispatch(params, wv, c, parray, n, rnd, wa, /**/ farray);
        break;
    }
    case WALL_VEL_V_HS: {
        WvelHS_v wv;
        wvel_get_view(w, /**/ &wv);
        stress_dispatch(params, wv, c, parray, n, rnd, wa, /**/ farray);
        break;
    }
    };
}

void wall_force_apply(const PairParams *params, const WvelStep *wv, const Coords *c, const PaArray *parray, int n, RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {    
    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaArray_v pav;        
        UC(pair_get_view_dpd_mirrored(params, &pv));
        parray_get_view(parray, &pav);
        wvel_dispatch(pv, wv, c, pav, n, rnd, wa, /**/ farray);
    }
    else {
        PairDPD pv;
        PaArray_v pav;
        UC(pair_get_view_dpd(params, &pv));
        parray_get_view(parray, &pav);
        wvel_dispatch(pv, wv, c, pav, n, rnd, wa, /**/ farray);
    }
}

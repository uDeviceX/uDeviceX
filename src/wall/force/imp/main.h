enum {
    TPP = 3 /* # threads per particle */
};

template<typename Par>
static void interactions(Par params, Wvel_v wv, const Coords *c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff) {
    Coords_v coordsv;
    coords_get_view(c, &coordsv);
    
    KL(wf_dev::force,
       (k_cnf(TPP*n)),
       (params, wv, coordsv, cloud, n, rnd_get(rnd), wa, /**/ (float*)ff));
}

void wall_force_apply(const PairParams *params, Wvel_v wv, const Coords *c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff) {
    PairDPD pv;
    UC(pair_get_view_dpd(params, &pv));
    interactions(pv, wv, c, cloud, n, rnd, wa, /**/ ff);
}

void wall_force_apply_color(const PairParams *params, Wvel_v wv, const Coords *c, Cloud cloud, int n, RNDunif *rnd, WallForce wa, /**/ Force *ff) {
    if (multi_solvent) {
        PairDPDCM pv;
        UC(pair_get_view_dpd_mirrored(params, &pv));
        interactions(pv, wv, c, cloud, n, rnd, wa, /**/ ff);
    }
    else {
        PairDPD pv;
        UC(pair_get_view_dpd(params, &pv));
        interactions(pv, wv, c, cloud, n, rnd, wa, /**/ ff);
    }
}

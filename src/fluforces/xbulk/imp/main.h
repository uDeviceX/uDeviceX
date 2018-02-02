enum {THR=128};

static void tbcloud_ini(int n, BCloud cloud, TBCloud *tc) {
    setup0((float4*) cloud.pp, 2*n, &tc->pp);
    if (multi_solvent) setup0(   (int*) cloud.cc,   n, &tc->cc);    
}

static void tbcloud_fin(TBCloud *tc) {
    destroy(&tc->pp);
    if (multi_solvent) destroy(&tc->cc);
}

template<typename Par>
void flocal_common(Par params, int3 L, int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff) {    
    float seed;
    TBCloud tc;
    if (n <= 0) return;
    seed = rnd_get(rnd);

    tbcloud_ini(n, cloud, &tc);
    
    KL(flocaldev::apply,
       (ceiln((n), THR), THR),
       (params, L, n, tc, start, seed, /**/ ff));

    tbcloud_fin(&tc);
}

void flocal(const PairParams *params, int3 L, int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff) {
    PairDPD pv;
    pair_get_view_dpd(params, &pv);
    flocal_common(pv, L, n, cloud, start, rnd, /**/ ff);
}

void flocal_colors(const PairParams *params, int3 L, int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff) {
    PairDPDC pv;
    pair_get_view_dpd_color(params, &pv);
    flocal_common(pv, L, n, cloud, start, rnd, /**/ ff);
}

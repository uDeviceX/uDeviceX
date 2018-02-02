enum {THR=128};

template<typename Par>
void oflocal_common(Par params, int3 L, int n, BCloud cloud, const int *start, RNDunif *rnd, /**/ Force *ff) {
    float seed;
    if (n <= 0) return;
    seed = rnd_get(rnd);
    KL(flocaldev::apply_unroll,
       (ceiln((n), THR), THR),
       (params, L, n, cloud, start, seed, /**/ ff));
}

static void tbcloud_ini(int n, BCloud cloud, TBCloud *tc) {
    setup0((float4*) cloud.pp, 2*n, &tc->pp);
    if (multi_solvent) setup0(   (int*) cloud.cc,   n, &tc->cc);    
}

static void tbcloud_fin(TBCloud *tc) {
    destroy(&tc->pp);
    if (multi_solvent) destroy(&tc->cc);
}

// try with textures
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


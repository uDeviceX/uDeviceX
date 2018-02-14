enum {THR=128};

static void tbarray_ini(int n, BPaArray parray, TBPaArray *ta) {
    setup0((float4*) parray.pp, 2*n, &ta->pp);
    if (multi_solvent) setup0(   (int*) parray.cc,   n, &ta->cc);    
}

static void tbarray_fin(TBPaArray *ta) {
    destroy(&ta->pp);
    if (multi_solvent) destroy(&ta->cc);
}

template<typename Par>
static void interactions(Par params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ Force *ff) {    
    float seed;
    TBPaArray ta;
    if (n <= 0) return;
    seed = rnd_get(rnd);

    tbarray_ini(n, parray, &ta);
    
    KL(flocaldev::apply,
       (ceiln((n), THR), THR),
       (params, L, n, ta, start, seed, /**/ ff));

    tbarray_fin(&ta);
}

void flocal(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ Force *ff) {
    PairDPD pv;
    pair_get_view_dpd(params, &pv);
    interactions(pv, L, n, parray, start, rnd, /**/ ff);
}

void flocal_color(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ Force *ff) {
    PairDPDC pv;
    pair_get_view_dpd_color(params, &pv);
    interactions(pv, L, n, parray, start, rnd, /**/ ff);
}

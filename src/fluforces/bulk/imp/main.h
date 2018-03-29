void flocal_push_pp(const float4 *pp, BPaArray *a) {
    a->pp = pp;
    a->colors = false;
}

void flocal_push_cc(const int *cc, BPaArray *a) {
    a->cc = cc;
    a->colors = true;
}

static void barray_get_view(BPaArray a, BPaArray_v *v) {
    v->pp = a.pp;
}

static void barray_get_view(BPaArray a, BPaCArray_v *v) {
    v->pp = a.pp;
    v->cc = a.cc;
}

static void tbarray_get_view(int n, BPaArray a, TBPaArray_v *v) {
    texo_setup(2*n, (float4*) a.pp, &v->pp);
}

static void tbarray_get_view(int n, BPaArray a, TBPaCArray_v *v) {
    texo_setup(2*n, (float4*) a.pp, &v->pp);
    texo_setup(n,      (int*) a.cc, &v->cc);    
}

static void tbarray_fin_view(TBPaArray_v *v) {
    texo_destroy(&v->pp);
}

static void tbarray_fin_view(TBPaCArray_v *v) {
    texo_destroy(&v->pp);
    texo_destroy(&v->cc);
}

template <typename Par, typename Parray, typename Farray>
static void interactions(Par params, int3 L, int n, Parray parray, const int *start, RNDunif *rnd, /**/ Farray farray) {    
    enum {THR=128};
    float seed;
    if (n <= 0) return;
    seed = rnd_get(rnd);
    
    KL(fbulk_dev::apply,
       (ceiln((n), THR), THR),
       (params, L, n, parray, start, seed, /**/ farray));
}

template <typename Par, typename Parray>
static void apply(Par params, int3 L, int n, Parray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, n, parray, start, rnd, /**/ farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, n, parray, start, rnd, /**/ farray_v);
    }
}

static void apply_color(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    PairDPDC pv;
    TBPaCArray_v parray_v;
        
    pair_get_view_dpd_color(params, &pv);
    tbarray_get_view(n, parray, &parray_v);    
        
    apply(pv, L, n, parray_v, start, rnd, /**/ farray);

    tbarray_fin_view(&parray_v);    
}

static void apply_grey(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    PairDPD pv;
    TBPaArray_v parray_v;
    pair_get_view_dpd(params, &pv);
    tbarray_get_view(n, parray, &parray_v);
        
    apply(pv, L, n, parray_v, start, rnd, /**/ farray);

    tbarray_fin_view(&parray_v); 
}

void flocal_apply(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    if (parray.colors)
        apply_color(params, L, n, parray, start, rnd, /**/ farray);
    else
        apply_grey(params, L, n, parray, start, rnd, /**/ farray);
}

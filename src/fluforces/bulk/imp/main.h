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
    setup0((float4*) a.pp, 2*n, &v->pp);
}

static void tbarray_get_view(int n, BPaArray a, TBPaCArray_v *v) {
    setup0((float4*) a.pp, 2*n, &v->pp);
    setup0(   (int*) a.cc,   n, &v->cc);    
}

static void tbarray_fin_view(TBPaArray_v *v) {
    destroy(&v->pp);
}

static void tbarray_fin_view(TBPaCArray_v *v) {
    destroy(&v->pp);
    destroy(&v->cc);
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

static void apply_color(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ FoArray farray) {
    PairDPDC pv;
    TBPaCArray_v parray_v;
    FoArray_v farray_v;
        
    pair_get_view_dpd_color(params, &pv);
    tbarray_get_view(n, parray, &parray_v);
    farray_get_view(&farray, &farray_v);
        
    interactions(pv, L, n, parray_v, start, rnd, /**/ farray_v);

    tbarray_fin_view(&parray_v);    
}

static void apply_grey(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ FoArray farray) {
    PairDPD pv;
    TBPaArray_v parray_v;
    FoArray_v farray_v;
    pair_get_view_dpd(params, &pv);
    tbarray_get_view(n, parray, &parray_v);
    farray_get_view(&farray, &farray_v);
        
    interactions(pv, L, n, parray_v, start, rnd, /**/ farray_v);

    tbarray_fin_view(&parray_v); 
}

void flocal_apply(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ Force *ff) {
    /* hack for now */
    FoArray farray;
    farray_push_ff(ff, &farray);
    
    if (parray.colors)
        apply_color(params, L, n, parray, start, rnd, /**/ farray);
    else
        apply_grey(params, L, n, parray, start, rnd, /**/ farray);
}

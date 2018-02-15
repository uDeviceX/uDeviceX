enum {
    PADDING = 16 /* warpsize / 2 */
};

template <typename Parray>
static void get_view_lfrag(const flu::LFrag26 f, LFrag_v26<Parray> *v) {
    Parray pav;
    LFrag_v<Parray> *vi;
    const flu::LFrag *fi;
    
    for (int i = 0; i < 26; ++i) {
        vi = &v->d[i];
        fi = &f.d[i];
        parray_get_view(&fi->parray, &pav);

        vi->parray = pav;
        vi->ii = fi->ii;
        vi->n  = fi->n;
    }
}

template <typename Parray>
static void get_view_rfrag(const flu::RFrag26 f, RFrag_v26<Parray> *v) {
    Parray pav;
    RFrag_v<Parray> *vi;
    const flu::RFrag *fi;
    
    for (int i = 0; i < 26; ++i) {
        vi = &v->d[i];
        fi = &f.d[i];
        parray_get_view(&fi->parray, &pav);

        vi->parray = pav;
        vi->start = fi->start;
        vi->dx = fi->dx;
        vi->dy = fi->dy;
        vi->dz = fi->dz;
        vi->xcells = fi->xcells;
        vi->ycells = fi->ycells;
        vi->zcells = fi->zcells;
        vi->type = fi->type;
    }
}

static bool is_colored(const flu::RFrag26 f) {
    return parray_is_colored(&(f.d[0].parray));
}

static int pad(int n) {
    return PADDING * ceiln(n, PADDING);    
}

template <typename Parray>
static void get_start(const LFrag_v<Parray> lfrags[26], /**/ int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i)
        start[i + 1] = start[i] + pad(lfrags[i].n);
}

template <typename Par, typename Parray, typename Farray>
static void interactions(Par params, int3 L, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ Farray farray) {
    int27 start;
    int n; /* number of threads */
    get_start(lfrags.d, /**/ start.d);
    n = start.d[26];
    KL(fhalo_dev::apply, (k_cnf(n)), (params, L, start, lfrags, rfrags, rrnd, /**/ farray));
}

template <typename Par, typename Parray>
static void apply(Par params, int3 L, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {

    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, lfrags, rfrags, rrnd, /**/ farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, lfrags, rfrags, rrnd, /**/ farray_v);
    }
}

static void apply_grey(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    PairDPD pv;
    LFrag_v26<PaArray_v> lfragsv;
    RFrag_v26<PaArray_v> rfragsv;

    pair_get_view_dpd(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);
    
    apply(pv, L, lfragsv, rfragsv, rrnd, /**/ farray);
}

static void apply_color(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    PairDPDC pv;
    LFrag_v26<PaCArray_v> lfragsv;
    RFrag_v26<PaCArray_v> rfragsv;

    pair_get_view_dpd_color(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);
    
    apply(pv, L, lfragsv, rfragsv, rrnd, /**/ farray);
}

void fhalo_apply(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    if (is_colored(rfrags))
        apply_color(params, L, lfrags, rfrags, rrnd, /**/ farray);
    else
        apply_grey(params, L, lfrags, rfrags, rrnd, /**/ farray);
}

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

template<typename Par, typename Parray>
static void interactions(Par params, int3 L, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    int27 start;
    int n; /* number of threads */
    get_start(lfrags.d, /**/ start.d);
    n = start.d[26];
    KL(fluforcesh_dev::force, (k_cnf(n)), (params, L, start, lfrags, rfrags, rrnd, /**/ ff));
}

void fhalo_apply(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    PairDPD pv;
    LFrag_v26<PaArray_v> lfragsv;
    RFrag_v26<PaArray_v> rfragsv;

    pair_get_view_dpd(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);
    
    interactions(pv, L, lfragsv, rfragsv, rrnd, /**/ ff);
}

void fhalo_apply_color(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    PairDPDC pv;
    LFrag_v26<PaCArray_v> lfragsv;
    RFrag_v26<PaCArray_v> rfragsv;

    pair_get_view_dpd_color(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);
    
    interactions(pv, L, lfragsv, rfragsv, rrnd, /**/ ff);
}

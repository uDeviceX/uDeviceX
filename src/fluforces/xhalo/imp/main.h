enum {
    PADDING = 16 /* warpsize / 2 */
};
      
static int pad(int n) {
    return PADDING * ceiln(n, PADDING);    
}

static void get_start(const flu::LFrag lfrags[26], /**/ int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i)
        start[i + 1] = start[i] + pad(lfrags[i].n);
}

template<typename Par>
static void interactions(Par params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    int27 start;
    int n; /* number of threads */
    get_start(lfrags.d, /**/ start.d);
    n = start.d[26];
    KL(dev::force, (k_cnf(n)), (params, L, start, lfrags, rfrags, rrnd, /**/ ff));
}

void fhalo(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    PairDPD pv;
    pair_get_view_dpd(params, &pv);
    interactions(pv, L, lfrags, rfrags, rrnd, /**/ ff);
}

void fhalo_color(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    PairDPDC pv;
    pair_get_view_dpd_color(params, &pv);
    interactions(pv, L, lfrags, rfrags, rrnd, /**/ ff);
}

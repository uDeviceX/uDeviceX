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

void interactions(int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    int27 start;
    int n; /* number of threads */
    get_start(lfrags.d, /**/ start.d);
    n = start.d[26];
    KL(dev::force, (k_cnf(n)), (L, start, lfrags, rfrags, rrnd, /**/ ff));
}

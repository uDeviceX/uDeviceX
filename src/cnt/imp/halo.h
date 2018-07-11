static int scan(int n, const int cc[], int ss[]) {
    int i, s;
    ss[0] = 0;
    for (i = s = 0; i < n; ++i)
        ss[i + 1] = (s += cc[i]);
    return ss[n];
}

static void halo_one_type(const PairParams *params, const Contact *c,
                          int nw, PaWrap *pw, FoWrap *fw,
                          Pap26 PP, Fop26 FF, const int *counts) {
    int i, n;
    const int *cellstarts;
    const uint *ids;
    int27 starts;
    PairDPDLJ pv;
    PaWrap pwi;
    FoWrap fwi;

    n = scan(26, counts, starts.d);    
    UC(pair_get_view_dpd_lj(params, &pv));

    for (i = 0; i < nw; ++i) {
        pwi = pw[i];
        fwi = fw[i];
        cellstarts = clists_get_ss(c->cells[i]);
        ids = clist_get_ids(c->cmap[i]);

        KL(cnt_dev::halo, (k_cnf(n)),
           (pv, c->L, rnd_get(c->rgen), starts,
            n, PP, cellstarts, ids, (const float2*) pwi.pp,
            /**/ FF, (float*) fwi.ff));        
    }
}

void cnt_halo(const PairParams *params, const Contact *cnt,
              int nw, PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff,
              const int *all_counts) {
    enum {NFRAGS = 26};
    Pap26 pp;
    Fop26 ff;
    int i, j, start[NFRAGS] = {0};
    const int *counts;
    
    for (i = 0; i < nw; i++) {
        counts = all_counts + NFRAGS * i;
        for (j = 0; j < NFRAGS; ++j) {
            pp.d[j] = all_pp.d[j] + start[j];
            ff.d[j] = all_ff.d[j] + start[j];
            start[j] += counts[j];
        }
        halo_one_type(params, cnt, nw, pw, fw, pp, ff, counts);
    }
}

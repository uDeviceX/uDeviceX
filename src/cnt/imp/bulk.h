void cnt_bulk(const PairParams *params, const Contact *c, int nw, PaWrap *pw, FoWrap *fw) {
    int i, j, self;
    float rnd;
    PairDPDLJ pv;
    const int *starts;
    const uint *ids;
    PaWrap pwi, pwj;
    FoWrap fwi, fwj;
    if (nw == 0) return;

    UC(pair_get_view_dpd_lj(params, &pv));
    
    for (i = 0; i < nw; ++i) {
        pwi = pw[i];
        fwi = fw[i];
        for (j = 0; j <= i; ++j) {
            self = (i == j);
            pwj = pw[j];
            fwj = fw[j];

            starts = clists_get_ss(c->cells[j]);
            ids    = clist_get_ids(c->cmap[j]);
            
            rnd = rnd_get(c->rgen);
            KL(cnt_dev::bulk, (k_cnf(3 * pwi.n)),
               (self, pv, c->L, 
                pwi.n, (const float2*) pwi.pp,
                starts, ids, (const float2*) pwj.pp,
                rnd, /**/ (float*) fwi.ff, (float*) fwj.ff));
        }
    }
}


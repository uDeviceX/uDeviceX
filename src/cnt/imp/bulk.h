static int get_id_inter(int a, int b) {
    int i, j;
    i = a > b ? a : b;
    j = a > b ? b : a;
    return j + i*(i+1)/2;
    
}

void cnt_bulk(const Contact *c, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw) {
    int i, j, self, id_prm;
    float rnd;
    const PairParams *prm;
    PairDPDLJ pv;
    const int *starts;
    const uint *ids;
    PaWrap pwi, pwj;
    FoWrap fwi, fwj;
    if (nw == 0) return;
    
    for (i = 0; i < nw; ++i) {
        pwi = pw[i];
        fwi = fw[i];
        for (j = 0; j <= i; ++j) {
            self = (i == j);
            pwj = pw[j];
            fwj = fw[j];

            starts = clists_get_ss(c->cells[j]);
            ids    = clist_get_ids(c->cmap[j]);

            id_prm = get_id_inter(i, j);

            prm = prms[id_prm];
            if (!prm) continue;
            UC(pair_get_view_dpd_lj(prm, &pv));
            
            rnd = rnd_get(c->rgen);
            KL(cnt_dev::bulk, (k_cnf(3 * pwi.n)),
               (self, pv, c->L, 
                pwi.n, (const float2*) pwi.pp,
                starts, ids, (const float2*) pwj.pp,
                rnd, /**/ (float*) fwi.ff, (float*) fwj.ff));
        }
    }
}


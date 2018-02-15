static float2pWraps convert(int nw, PaWrap *pw) {
    float2pWraps w = {0};
    for (int i = 0; i < nw; ++i)
        w.d[i] = (const float2 *) pw[i].pp;
    return w;
}

static ForcepWraps convert(int nw, FoWrap *fw) {
    ForcepWraps w = {0};
    for (int i = 0; i < nw; ++i)
        w.d[i] = (float *) fw[i].ff;
    return w;
}

void cnt_bulk(const PairParams *params, const Contact *c, int nw, PaWrap *pw, FoWrap *fw) {
    float rnd;
    PairDPDLJ pv;
    if (nw == 0) return;
    float2pWraps lpp = convert(nw, pw);
    ForcepWraps  lff = convert(nw, fw);

    UC(pair_get_view_dpd_lj(params, &pv));
    
    for (int i = 0; i < nw; ++i) {
        PaWrap pit = pw[i];
        FoWrap fit = fw[i];
        rnd = rnd_get(c->rgen);
        KL(cnt_dev::bulk, (k_cnf(3 * pit.n)),
           (pv, c->L, c->cells.starts, clist_get_ids(c->cmap), pit.n, (const float2*)pit.pp, lpp, rnd, i, /**/ lff, (float*)fit.ff));
    }
}


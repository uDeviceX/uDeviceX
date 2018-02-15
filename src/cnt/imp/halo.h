void cnt_halo(const PairParams *params, const Contact *c, int nw, PaWrap *pw, FoWrap *fw, Pap26 PP, Fop26 FF, int counts[26]) {
    int i, s, n;
    int27 starts;
    PairDPDLJ pv;
    
    starts.d[0] = 0;
    for (i = s = 0; i < 26; ++i) starts.d[i + 1] = (s += counts[i]);
    n = starts.d[26];

    auto lpp = convert(nw, pw);
    auto lff = convert(nw, fw);

    UC(pair_get_view_dpd_lj(params, &pv));
    
    KL(cnt_dev::halo, (k_cnf(n)),
       (pv, c->L, c->cells.starts, clist_get_ids(c->cmap), rnd_get(c->rgen), starts, lpp, n, PP, /**/ lff, FF));
}

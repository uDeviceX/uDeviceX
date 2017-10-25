void halo(int nw, PaWrap *pw, Pap26 PP, Fop26 FF, int counts[26]) {
    int i, s, n;
    int27 starts;
    starts.d[0] = 0;
    for (i = s = 0; i < 26; ++i) starts.d[i + 1] = (s += counts[i]);
    n = starts.d[26];

    auto lpp = convert(nw, pw);
    
    KL(dev::halo, (k_cnf(n)), (g::rgen->get_float(), starts, lpp, n, PP, /**/ FF));
}

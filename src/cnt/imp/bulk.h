static float2pWraps convert(int nw, PaWrap *pw) {
    float2pWraps w = {0};
    for (int i = 0; i < nw; ++i)
        w.d[i] = (const float2 *) pw[i].pp;
    return w;
}

void bulk(int nw, PaWrap *pw, FoWrap *fw) {
    float rnd;
    if (nw == 0) return;
    float2pWraps w = convert(nw, pw);    

    for (int i = 0; i < nw; ++i) {
        PaWrap pit = pw[i];
        FoWrap fit = fw[i];
        rnd = g::rgen->get_float();
        KL(dev::bulk, (k_cnf(3 * pit.n)),
           (pit.n, (const float2*)pit.pp, w, rnd, i, (float*)fit.ff));
    }
}


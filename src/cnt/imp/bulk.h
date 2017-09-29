void bulk(int nw, PaWrap *pw, FoWrap *fw) {
    float rnd;
    if (nw == 0) return;
    for (int i = 0; i < nw; ++i) {
        PaWrap pit = pw[i];
        FoWrap fit = fw[i];
        rnd = g::rgen->get_float();
        KL(dev::bulk, (k_cnf(3 * pit.n)),
           ((float2*)pit.pp, pit.n, g::entries->S, rnd, i, (float*)fit.ff));
    }
}


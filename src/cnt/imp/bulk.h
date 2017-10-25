void bulk(int nw, PaWrap *pw, FoWrap *fw) {
    float rnd;
    if (nw == 0) return;
    for (int i = 0; i < nw; ++i) {
        PaWrap pit = pw[i];
        FoWrap fit = fw[i];
        rnd = g::rgen->get_float();
        KL(dev::bulk, (k_cnf(3 * pit.n)),
           (pit.n, (const float2*)pit.pp, rnd, i, (float*)fit.ff));
    }
}


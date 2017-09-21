void bulk(std::vector<PaWrap> pwr, std::vector<FoWrap> fwr) {
    float rnd;
    if (pwr.size() == 0) return;
    for (int i = 0; i < (int) pwr.size(); ++i) {
        PaWrap pit = pwr[i];
        FoWrap fit = fwr[i];
        rnd = rgen->get_float();
        KL(dev::bulk, (k_cnf(3 * pit.n)),
           ((float2*)pit.pp, pit.n, entries->S, pwr.size(), rnd, i, (float*)fit.ff));
    }
}


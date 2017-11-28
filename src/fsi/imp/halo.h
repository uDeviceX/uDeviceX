void halo(Pap26 PP, Fop26 FF, int counts[26]) {
    int i, n, s;
    int27 starts;
    const Cloud cloud = wo->c;
    // setup(wo->starts);
    starts.d[0] = 0;
    for (i = s = 0; i < 26; ++i) starts.d[i + 1] = (s += counts[i]);
    n = starts.d[26];

    KL(dev::halo, (k_cnf(n)), (wo->starts, starts, PP, FF, cloud, n, wo->n, rgen->get_float(), /**/ (float*)wo->ff));
}

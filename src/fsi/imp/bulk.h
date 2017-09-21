static void bulk0(PaWrap *pw, FoWrap *fw, hforces::Cloud cloud) {
    int n0, n1;
    float rnd;
    const Particle *ppA = pw->pp;
    rnd = rgen->get_float();
    n0 = pw->n;
    n1 = wo->n;
    KL(dev::bulk, (k_cnf(3*n0)), ((float*)ppA, cloud, \
                                  n0, n1, \
                                  rnd, (float*)fw->ff, (float*)wo->ff));
}

void bulk(std::vector<PaWrap> pwr, std::vector<FoWrap> fwr) {
    int i, n;
    PaWrap *pw; /* wrap */
    FoWrap *fw; /* wrap */
    n = pwr.size();
    pw = pwr.data();
    fw = fwr.data();

    if (n == 0) return;
    setup(wo->starts);
    for (i = 0; i < n; i++) bulk0(pw++, fw++, wo->c);
}

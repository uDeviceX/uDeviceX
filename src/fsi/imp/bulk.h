static void bulk0(const int *start, PaWrap *pw, FoWrap *fw, Cloud cloud) {
    int n0, n1;
    float rnd;
    const Particle *ppA = pw->pp;
    rnd = rgen->get_float();
    n0 = pw->n;
    n1 = wo->n;
    KL(dev::bulk, (k_cnf(3*n0)), (start, (float*)ppA, cloud,     \
                                  n0, n1, \
                                  rnd, (float*)fw->ff, (float*)wo->ff));
}

void bulk(int nw, PaWrap *pw, FoWrap *fw) {
    int i;

    if (nw == 0) return;
    // setup(wo->starts);
    for (i = 0; i < nw; i++) bulk0(wo->starts, pw++, fw++, wo->c);
}

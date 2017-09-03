static void bulk0(ParticlesWrap *w, const Particle *ppB) {
    int n0, n1;
    float rnd;
    const Particle *ppA = w->p;
    rnd = rgen->get_float();
    n0 = w->n;
    n1 = wo->n;
    KL(dev::bulk, (k_cnf(3*n0)), ((float*)ppA, (float*)ppB, n0, n1, rnd, (float*)w->f, (float*)wo->f));
}

void bulk(std::vector<ParticlesWrap> wr) {
    int i, n;
    ParticlesWrap *w; /* wrap */
    n = wr.size();
    w = wr.data();

    if (n == 0) return;
    setup(wo->cellsstart);
    for (i = 0; i < n; i++) bulk0(w++, wo->p);
}

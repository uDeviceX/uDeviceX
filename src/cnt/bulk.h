void bulk(std::vector<ParticlesWrap> wr) {
    float rnd;
    if (wr.size() == 0) return;
    for (int i = 0; i < (int) wr.size(); ++i) {
        ParticlesWrap it = wr[i];
        rnd = rgen->get_float();
        KL(dev::bulk, (k_cnf(3 * it.n)),
           ((float2*)it.p, it.n, entries->S, wr.size(), rnd, i, (float*)it.f));
    }
}


void halo(ParticlesWrap halos[26], Pap26 PP, Fop26 FF, int counts[26]) {
    int i, s, n;
    int counts[26], starts[27];
    Force *ff[26];
    const Particle *pp[26];
    for (i = 0; i < 26; ++i) counts[i] = halos[i].n;
    starts[0] = 0;
    for (i = s = 0; i < 26; ++i) starts[i + 1] = (s += halos[i].n);
    n = starts[26];
    for (i = 0; i < 26; ++i) pp[i] = halos[i].p;
    for (i = 0; i < 26; ++i) ff[i] = halos[i].f;

    CC(cudaMemcpyToSymbolAsync(dev::g::counts, counts, 26*sizeof(counts[0]), 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::starts, starts, sizeof(starts), 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::pp, PP.d, 26*sizeof(PP.d[0]), 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::ff, FF.d, 26*sizeof(FF.d[0]), 0, H2D));
    KL(dev::halo, (k_cnf(n)), (n, rgen->get_float()));
}

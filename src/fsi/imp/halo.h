void halo(ParticlesWrap halos[26], Pap26 PP, Fop26 FF, int nn[26]) {
    int i, n, s;
    int counts[26], starts[27];
    Force *ff[26];
    const Particle *pp[26];
    const Particle *ppB = wo->p;
    setup(ppB, wo->n, wo->cellsstart);
    for (i = 0; i < 26; ++i) counts[i] = halos[i].n;
    starts[0] = 0;
    for (i = s = 0; i < 26; ++i) starts[i + 1] = (s += halos[i].n);
    n = starts[26];
    for (i = 0; i < 26; ++i) pp[i] = halos[i].p;
    for (i = 0; i < 26; ++i) ff[i] = halos[i].f;

    CC(cudaMemcpyToSymbolAsync(dev::g::counts, counts, sizeof(counts), 0, H2D));    
    CC(cudaMemcpyToSymbolAsync(dev::g::starts, starts, sizeof(starts), 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::pp, PP.d, sizeof(PP.d), 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::ff, FF.d, sizeof(FF.d), 0, H2D));
    KL(dev::halo, (k_cnf(n)), ((float*)ppB, n, wo->n, rgen->get_float(), /**/ (float*)wo->f));
}

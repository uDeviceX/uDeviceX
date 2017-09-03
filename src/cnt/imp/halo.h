void halo(ParticlesWrap halos[26]) {
    int i, s, n;
    int recvpackcount[26], recvpackstarts_padded[27];
    Force *ff[26];
    const Particle *recvpackstates[26];
    n = 0;
    for (i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;
    CC(cudaMemcpyToSymbolAsync(dev::g::counts, recvpackcount, sizeof(recvpackcount), 0, H2D));
    recvpackstarts_padded[0] = 0;
    for (i = s = 0; i < 26; ++i)
        recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));
    n = recvpackstarts_padded[26];
    CC(cudaMemcpyToSymbolAsync
       (dev::g::starts, recvpackstarts_padded,
        sizeof(recvpackstarts_padded), 0, H2D));
    
    for (i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;
    
    CC(cudaMemcpyToSymbolAsync(dev::g::pp, recvpackstates,
                               sizeof(recvpackstates), 0,
                               H2D));
    for (i = 0; i < 26; ++i) ff[i] = halos[i].f;
    CC(cudaMemcpyToSymbolAsync(dev::g::ff, ff, sizeof(ff), 0, H2D));
    KL(dev::halo, (k_cnf(n)), (n, rgen->get_float()));
}

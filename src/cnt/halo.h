namespace cnt {
void halo(ParticlesWrap halos[26]) {
    int n = 0;
    {
        int recvpackcount[26], recvpackstarts_padded[27];

        for (int i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;

        CC(cudaMemcpyToSymbolAsync(k_cnt::g::counts, recvpackcount,
                                   sizeof(recvpackcount), 0, H2D));
        recvpackstarts_padded[0] = 0;
        for (int i = 0, s = 0; i < 26; ++i)
        recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));
        n = recvpackstarts_padded[26];
        CC(cudaMemcpyToSymbolAsync
           (k_cnt::g::starts, recvpackstarts_padded,
            sizeof(recvpackstarts_padded), 0, H2D));

        const Particle *recvpackstates[26];
        for (int i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;

        CC(cudaMemcpyToSymbolAsync(k_cnt::g::pp, recvpackstates,
                                   sizeof(recvpackstates), 0,
                                   H2D));
        Force *ff[26];
        for (int i = 0; i < 26; ++i) ff[i] = halos[i].f;
        CC(cudaMemcpyToSymbolAsync(k_cnt::g::ff, ff, sizeof(ff), 0, H2D));
    }
    KL(k_cnt::halo, (k_cnf(n)), (n, entries->S, no, rgen->get_float()));
}
}

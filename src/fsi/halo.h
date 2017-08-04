namespace fsi {
void halo(ParticlesWrap halos[26]) {
    setup(wsolvent->p, wsolvent->n, wsolvent->cellsstart, wsolvent->cellscount);
    int nremote_padded = 0;
    int recvpackcount[26], recvpackstarts_padded[27];

    for (int i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;
    CC(cudaMemcpyToSymbolAsync(k_fsi::g::counts, recvpackcount, sizeof(recvpackcount), 0, H2D));

    recvpackstarts_padded[0] = 0;
    for (int i = 0, s = 0; i < 26; ++i)
        recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));

    nremote_padded = recvpackstarts_padded[26];

    CC(cudaMemcpyToSymbolAsync(
                               k_fsi::g::starts, recvpackstarts_padded,
                               sizeof(recvpackstarts_padded), 0, H2D));

    const Particle *recvpackstates[26];

    for (int i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;

    CC(cudaMemcpyToSymbolAsync(k_fsi::g::pp, recvpackstates,
                               sizeof(recvpackstates), 0,
                               H2D));

    Force *ff[26];
    for (int i = 0; i < 26; ++i) ff[i] = halos[i].f;
    CC(cudaMemcpyToSymbolAsync(k_fsi::g::ff, ff, sizeof(ff), 0, H2D));

    if (nremote_padded)
        k_fsi::halo<<<k_cnf(nremote_padded)>>>
            (nremote_padded, wsolvent->n, local_trunk->get_float(), /**/ (float *)wsolvent->f);
}
}

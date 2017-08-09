namespace fsi {
void halo(ParticlesWrap halos[26]) {
    setup(wsolvent->p, wsolvent->n, wsolvent->cellsstart, wsolvent->cellscount);
    int n = 0;
    int counts[26], starts[27];
    for (int i = 0; i < 26; ++i) counts[i] = halos[i].n;
    CC(cudaMemcpyToSymbolAsync(k_fsi::g::counts, counts, sizeof(counts), 0, H2D));
    starts[0] = 0;
    for (int i = 0, s = 0; i < 26; ++i) starts[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));
    n = starts[26];

    CC(cudaMemcpyToSymbolAsync(k_fsi::g::starts, starts, sizeof(starts), 0, H2D));
    const Particle *pp[26];
    for (int i = 0; i < 26; ++i) pp[i] = halos[i].p;
    CC(cudaMemcpyToSymbolAsync(k_fsi::g::pp, pp, sizeof(pp), 0, H2D));
    Force *ff[26];
    for (int i = 0; i < 26; ++i) ff[i] = halos[i].f;
    CC(cudaMemcpyToSymbolAsync(k_fsi::g::ff, ff, sizeof(ff), 0, H2D));

    if (n)
        KL(k_fsi::halo, (k_cnf(n)), (n, wsolvent->n, rgen->get_float(), /**/ (float*)wsolvent->f));
}
}

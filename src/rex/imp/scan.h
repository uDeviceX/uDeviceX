namespace sub {
void scanA(ParticlesWrap *w, int nw, rex::TicketPack tp) {
    int i, n;
    int *o0, *o1; /* offsets */
    int *c;       /* counts */
    int *s;       /* starts */
    for (i = 0; i < nw; ++i) {
        const Particle *pp = w[i].p;
        o0 = tp.offsets + 26 *  i;
        o1 = tp.offsets + 26 * (i + 1);
        c  = tp.counts  + 26 *  i;
        s  = tp.starts  + 27 *  i;
        n = w[i].n;
        if (n) {
            CC(cudaMemcpyToSymbolAsync(dev::g::offsets, o0, sizeof(int) * 26, 0, D2D));
            KL(dev::scatter, (k_cnf(n)), ((float2*)pp, o0, n, /**/ c));
        }
        KL(dev::scanA, (1, 32), (c, o0, /**/ o1, s));
    }
}

void scanB(int nw, rex::TicketPack tp) {
    KL(dev::scanB, (1, 32), (tp.offsets + 26 * nw, /**/ tp.tstarts));
}
}

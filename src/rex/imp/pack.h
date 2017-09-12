namespace sub {
void clear(int nw, rex::TicketPack tp) {
    CC(cudaMemsetAsync(tp.starts,  0, sizeof(int) * 27 *  nw));
    CC(cudaMemsetAsync(tp.counts,  0, sizeof(int) * 26 *  nw));
    CC(cudaMemsetAsync(tp.offsets, 0, sizeof(int) * 26 * (nw + 1)));
}

static int i2max(int i) { /* fragment id to maximum size */
    return MAX_OBJ_DENSITY*frag_ncell(i);
}

void clear_forces(const int counts[26], /**/ Fop26 FF) {
    int i, c;
    for (i = 0; i < 26; ++i) {
        c = counts[i];
        CC(cudaMemsetAsync(FF.d[i], 0, sizeof(Force) * c));
    }
}

static void pack0(ParticlesWrap *w, rex::TicketPack tp, int i, /**/ Particle *buf) {
    int *o, *c, *s;
    const Particle *pp = w[i].p;
    o = tp.offsets + 26 *  i;
    c = tp.counts  + 26 *  i;
    s = tp.starts  + 27 *  i;
    CC(cudaMemcpyToSymbolAsync(dev::g::offsets, o, sizeof(int) * 26, 0, D2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::counts,  c, sizeof(int) * 26, 0, D2D));
    CC(cudaMemcpyToSymbolAsync(dev::g::starts,  s, sizeof(int) * 27, 0, D2D));
    KL(dev::pack, (14 * 16, 128), ((float2*)pp, /**/ (float2*)buf));
}

void pack(ParticlesWrap *w, int nw, rex::TicketPack tp, Particle *buf) {
    int i, n;
    CC(cudaMemcpyToSymbolAsync(dev::g::tstarts, tp.tstarts, sizeof(int) * 27, 0, D2D));
    for (i = 0; i < nw; ++i) {
        n = w[i].n;
        if (n) pack0(w, tp, i, /**/ buf);
    }
}

}

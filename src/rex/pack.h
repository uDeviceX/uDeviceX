namespace rex {
void clear(int nw, x::TicketPack tp) {
    CC(cudaMemsetAsync(tp.starts,  0, sizeof(int) * 27 *  nw));
    CC(cudaMemsetAsync(tp.counts,  0, sizeof(int) * 26 *  nw));
    CC(cudaMemsetAsync(tp.offsets, 0, sizeof(int) * 26 * (nw + 1)));
}

static void pack0(std::vector<ParticlesWrap> w, x::TicketPack tp, int i, /**/ Particle *buf) {
    int *o, *c, *s;
    const Particle *pp = w[i].p;
    o = tp.offsets + 26 *  i;
    c = tp.counts  + 26 *  i;
    s = tp.starts  + 27 *  i;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::offsets, o, sizeof(int) * 26, 0, D2D));
    CC(cudaMemcpyToSymbolAsync(k_rex::g::counts,  c, sizeof(int) * 26, 0, D2D));
    CC(cudaMemcpyToSymbolAsync(k_rex::g::starts,  s, sizeof(int) * 27, 0, D2D));
    KL(k_rex::pack, (14 * 16, 128), ((float2*)pp, /**/ (float2*)buf));
}

void pack(std::vector<ParticlesWrap> w, int nw, x::TicketPack tp, Particle *buf) {
    int i, n;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::tstarts, tp.tstarts, sizeof(int) * 27, 0, D2D));
    for (i = 0; i < nw; ++i) {
        n = w[i].n;
        if (n) pack0(w, tp, i, /**/ buf);
    }
}

}

namespace sub {
void unpack(ParticlesWrap *w, int nw, rex::TicketPack tp) {
    int i, n;
    Force *ff;
    int *o, *c, *s;
    for (i = 0; i < nw; ++i) {
        n = w[i].n;
        if (n) {
            ff = w[i].f;
            o = tp.offsets + 26 *  i;
            s = tp.starts  + 27 *  i;
            c = tp.counts  + 26 *  i;
            CC(cudaMemcpyToSymbolAsync(dev::g::offsets, o, sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(dev::g::starts,  s, sizeof(int) * 27, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(dev::g::counts,  c, sizeof(int) * 26, 0, D2D));
            KL(dev::unpack, (16 * 14, 128), /**/ ((float*)ff));
        }
    }
}
}

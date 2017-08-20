namespace rex {
void unpack(std::vector<ParticlesWrap> w0, int nw, x::TicketPack tp) {
    ParticlesWrap* w = w0.data();
    
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
            CC(cudaMemcpyToSymbolAsync(k_rex::g::offsets, o, sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::starts,  s, sizeof(int) * 27, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::counts,  c, sizeof(int) * 26, 0, D2D));
            KL(k_rex::unpack, (16 * 14, 128), /**/ ((float*)ff));
        }
    }
}
}

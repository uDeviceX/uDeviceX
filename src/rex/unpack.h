namespace rex {
void unpack(std::vector<ParticlesWrap> w, int nw, x::TicketPack tp) {
    int i, n;
    Force *ff;
    for (i = 0; i < nw; ++i) {
        n = w[i].n;
        if (n) {
            ff = w[i].f;
            CC(cudaMemcpyToSymbolAsync(k_rex::g::starts,  tp.starts  + 27 * i, sizeof(int) * 27, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::counts,  tp.counts  + 26 * i, sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::offsets, tp.offsets + 26 * i, sizeof(int) * 26, 0, D2D));
            k_rex::unpack<<<16 * 14, 128>>>(/**/ (float*)ff);
        }
    }
}
}

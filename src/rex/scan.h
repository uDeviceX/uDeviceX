namespace rex {
void scanA(std::vector<ParticlesWrap> w, int nw, x::TicketPack tp) {
    int i;
    k_rex::ini<<<1, 1>>>();
    for (i = 0; i < nw; ++i) {
        ParticlesWrap it = w[i];
        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::g::offsets, tp.offsets + 26 * i, sizeof(int) * 26, 0, D2D));
            k_rex::scatter<<<k_cnf(it.n)>>>((float2*)it.p, it.n, /**/ tp.counts + i * 26);
        }
        k_rex::scanA<<<1, 32>>>(tp.counts + i * 26, tp.offsets + 26 * i, /**/ tp.offsets + 26 * (i + 1), tp.starts + i * 27);
    }
}

void scanB(std::vector<ParticlesWrap> w, x::TicketPack tp) {
    CC(cudaMemcpyAsync(tp.offsets_hst, tp.offsets + 26 * w.size(), sizeof(int) * 26, H2H));
    k_rex::scanB<<<1, 32>>>(tp.offsets + 26 * w.size(), /**/ tp.tstarts);
}
}

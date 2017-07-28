namespace rex {
void pack_p(int nw, x::TicketPack tp) {
    tp.offsets->resize(26 * (nw + 1));
}

void pack_clear(int nw, x::TicketPack tp) {
    if (tp.offsets->S) CC(cudaMemsetAsync(tp.offsets->D, 0, sizeof(int) * tp.offsets->S));
    CC(cudaMemsetAsync(tp.starts, 0, sizeof(int) * 27 * nw));
    CC(cudaMemsetAsync(tp.counts, 0, sizeof(int) * 26 * nw));
}

void scanA(std::vector<ParticlesWrap> w, x::TicketPack tp) {
    k_rex::ini<<<1, 1>>>();
    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];
        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::g::offsets, tp.offsets->D + 26 * i, sizeof(int) * 26, 0, D2D));
            k_rex::scatter<<<k_cnf(it.n)>>>((float2 *)it.p, it.n, /**/ tp.counts + i * 26);
        }
        k_rex::scanA<<<1, 32>>>(tp.counts + i * 26, tp.offsets->D + 26 * i, /**/ tp.offsets->D + 26 * (i + 1), tp.starts + i * 27);
    }
}

void scanB(std::vector<ParticlesWrap> w, x::TicketPack tp) {
    CC(cudaMemcpyAsync(tp.offsets_hst->D, tp.offsets->D + 26 * w.size(), sizeof(int) * 26, H2H));
    k_rex::scanB<<<1, 32>>>(tp.offsets->D + 26 * w.size(), /**/ tp.tstarts->D);
}

void pack_attempt(std::vector<ParticlesWrap> w, x::TicketPack tp) {
    CC(cudaMemcpyAsync(tp.tstarts_hst->D, tp.tstarts->D, sizeof(int) * 27, H2H));
    CC(cudaMemcpyToSymbolAsync(k_rex::g::tstarts, tp.tstarts->D, sizeof(int) * 27, 0, D2D));
    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];
        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::g::offsets, tp.offsets->D + 26 * i, sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::counts, tp.counts + 26 * i, sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::g::starts, tp.starts + 27 * i, sizeof(int) * 27, 0, D2D));
            k_rex::pack<<<14 * 16, 128>>>((float2 *)it.p, /**/ (float2*)packbuf->D);
        }
    }
}

}

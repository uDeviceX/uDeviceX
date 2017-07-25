namespace rex {
void _pack_attempt(std::vector<ParticlesWrap> w, x::TicketPack tp) {


    if (tp.packscount->S)
        CC(cudaMemsetAsync(tp.packscount->D, 0, sizeof(int) * tp.packscount->S));

    if (tp.packsoffset->S)
        CC(cudaMemsetAsync(tp.packsoffset->D, 0, sizeof(int) * tp.packsoffset->S));

    if (tp.packsstart->S)
        CC(cudaMemsetAsync(tp.packsstart->D, 0, sizeof(int) * tp.packsstart->S));

    k_rex::ini<<<1, 1>>>();
    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];
        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, tp.packsoffset->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));
            k_rex::scatter<<<k_cnf(it.n)>>>((float2 *)it.p, it.n, /**/ tp.packscount->D + i * 26);
        }
        k_rex::scanA<<<1, 32>>>(tp.packscount->D + i * 26, tp.packsoffset->D + 26 * i,
                               /**/ tp.packsoffset->D + 26 * (i + 1), tp.packsstart->D + i * 27);
    }

    CC(cudaMemcpyAsync(tp.host_packstotalcount->D,
                       tp.packsoffset->D + 26 * w.size(), sizeof(int) * 26,
                       H2H));

    k_rex::scanB<<<1, 32>>>(tp.packsoffset->D + 26 * w.size(), /**/ tp.packstotalstart->D);

    CC(cudaMemcpyAsync(tp.host_packstotalstart->D, tp.packstotalstart->D,
                       sizeof(int) * 27, H2H));

    CC(cudaMemcpyToSymbolAsync(k_rex::cbases, tp.packstotalstart->D,
                               sizeof(int) * 27, 0, D2D));
    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];

        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, tp.packsoffset->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::ccounts, tp.packscount->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::cpaddedstarts, tp.packsstart->D + 27 * i,
                                       sizeof(int) * 27, 0, D2D));
            k_rex::pack<<<14 * 16, 128>>>((float2 *)it.p, /**/ (float2 *)packbuf->D);
        }
    }
}

void pack_p(int n, x::TicketPack tp) {
    tp.packscount->resize(26 * n);
    tp.packsoffset->resize(26 * (n + 1));
    tp.packsstart->resize(27 * n);
}

}

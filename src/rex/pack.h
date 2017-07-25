namespace rex {
void _pack_attempt(std::vector<ParticlesWrap> w) {


    if (packscount->S)
    CC(cudaMemsetAsync(packscount->D, 0, sizeof(int) * packscount->S));

    if (packsoffset->S)
    CC(cudaMemsetAsync(packsoffset->D, 0, sizeof(int) * packsoffset->S));

    if (packsstart->S)
    CC(cudaMemsetAsync(packsstart->D, 0, sizeof(int) * packsstart->S));

    k_rex::ini<<<1, 1>>>();
    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];
        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, packsoffset->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));
            k_rex::scatter<<<k_cnf(it.n)>>>((float2 *)it.p, it.n, /**/ packscount->D + i * 26);
        }
        k_rex::scan<<<1, 32>>>(packscount->D + i * 26, packsoffset->D + 26 * i,
                               /**/ packsoffset->D + 26 * (i + 1), packsstart->D + i * 27);
    }

    CC(cudaMemcpyAsync(host_packstotalcount->D,
                       packsoffset->D + 26 * w.size(), sizeof(int) * 26,
                       H2H));

    k_rex::scan<<<1, 32>>>(packsoffset->D + 26 * w.size(), NULL, /**/ NULL, packstotalstart->D);

    CC(cudaMemcpyAsync(host_packstotalstart->D, packstotalstart->D,
                       sizeof(int) * 27, H2H));

    CC(cudaMemcpyToSymbolAsync(k_rex::cbases, packstotalstart->D,
                               sizeof(int) * 27, 0, D2D));
    for (int i = 0; i < (int) w.size(); ++i) {
        ParticlesWrap it = w[i];

        if (it.n) {
            CC(cudaMemcpyToSymbolAsync(k_rex::coffsets, packsoffset->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::ccounts, packscount->D + 26 * i,
                                       sizeof(int) * 26, 0, D2D));
            CC(cudaMemcpyToSymbolAsync(k_rex::cpaddedstarts,
                                       packsstart->D + 27 * i, sizeof(int) * 27, 0,
                                       D2D));
            k_rex::pack<<<14 * 16, 128>>>((float2 *)it.p, i, /**/ (float2 *)packbuf->D);
        }
    }
}

void pack_p(int n) {
    packscount->resize(26 * n);
    packsoffset->resize(26 * (n + 1));
    packsstart->resize(27 * n);
}

}

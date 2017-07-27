namespace cnt {
void bind(const int *const cellsstart, const int *const cellentries,
          const int ncellentries, std::vector<ParticlesWrap> wsolutes) {
    size_t textureoffset = 0;

    if (ncellentries)
        CC(cudaBindTexture(&textureoffset, &k_cnt::texCellEntries, cellentries,
                           &k_cnt::texCellEntries.channelDesc,
                           sizeof(int) * ncellentries));
    int ncells = XS * YS * ZS;
    CC(cudaBindTexture(&textureoffset, &k_cnt::texCellsStart, cellsstart,
                       &k_cnt::texCellsStart.channelDesc, sizeof(int) * ncells));
    int n = wsolutes.size();

    int ns[n];
    float2 *ps[n];
    float *fs[n];

    for (int i = 0; i < n; ++i) {
        ns[i] = wsolutes[i].n;
        ps[i] = (float2 *)wsolutes[i].p;
        fs[i] = (float *)wsolutes[i].f;
    }

    CC(cudaMemcpyToSymbolAsync(k_cnt::cnsolutes, ns, sizeof(int) * n, 0,
                               H2D));
    CC(cudaMemcpyToSymbolAsync(k_cnt::csolutes, ps, sizeof(float2 *) * n, 0,
                               H2D));
    CC(cudaMemcpyToSymbolAsync(k_cnt::csolutesacc, fs, sizeof(float *) * n, 0,
                               H2D));
}
}

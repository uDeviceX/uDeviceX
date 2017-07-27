namespace k_cnt {
void setup() {
    texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
    texCellsStart.filterMode = cudaFilterModePoint;
    texCellsStart.mipmapFilterMode = cudaFilterModePoint;
    texCellsStart.normalized = 0;

    texCellEntries.channelDesc = cudaCreateChannelDesc<int>();
    texCellEntries.filterMode = cudaFilterModePoint;
    texCellEntries.mipmapFilterMode = cudaFilterModePoint;
    texCellEntries.normalized = 0;
}

void bind(const int *const cellsstart, const int *const cellentries,
          const int ncellentries, std::vector<ParticlesWrap> wsolutes) {
    size_t textureoffset = 0;

    if (ncellentries)
    CC(cudaBindTexture(&textureoffset, &texCellEntries, cellentries,
                       &texCellEntries.channelDesc,
                       sizeof(int) * ncellentries));
    int ncells = XS * YS * ZS;
    CC(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart,
                       &texCellsStart.channelDesc, sizeof(int) * ncells));
    int n = wsolutes.size();

    int ns[n];
    float2 *ps[n];
    float *fs[n];

    for (int i = 0; i < n; ++i) {
        ns[i] = wsolutes[i].n;
        ps[i] = (float2 *)wsolutes[i].p;
        fs[i] = (float *)wsolutes[i].f;
    }

    CC(cudaMemcpyToSymbolAsync(cnsolutes, ns, sizeof(int) * n, 0,
                               H2D));
    CC(cudaMemcpyToSymbolAsync(csolutes, ps, sizeof(float2 *) * n, 0,
                               H2D));
    CC(cudaMemcpyToSymbolAsync(csolutesacc, fs, sizeof(float *) * n, 0,
                               H2D));
}
}

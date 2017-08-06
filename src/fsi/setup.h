namespace fsi {
static void setup_first() {
    k_fsi::texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
    k_fsi::texCellsStart.filterMode = cudaFilterModePoint;
    k_fsi::texCellsStart.mipmapFilterMode = cudaFilterModePoint;
    k_fsi::texCellsStart.normalized = 0;

    k_fsi::texCellsCount.channelDesc = cudaCreateChannelDesc<int>();
    k_fsi::texCellsCount.filterMode = cudaFilterModePoint;
    k_fsi::texCellsCount.mipmapFilterMode = cudaFilterModePoint;
    k_fsi::texCellsCount.normalized = 0;

    k_fsi::texSolventParticles.channelDesc = cudaCreateChannelDesc<float2>();
    k_fsi::texSolventParticles.filterMode = cudaFilterModePoint;
    k_fsi::texSolventParticles.mipmapFilterMode = cudaFilterModePoint;
    k_fsi::texSolventParticles.normalized = 0;
}

void setup(const Particle *const solvent, int n, const int *const cellsstart, const int *const cellscount) {
    size_t offset;
    int nc;
    if (firsttime) {
        setup_first();
        firsttime = false;
    }

    offset = 0;
    if (n)
        CC(cudaBindTexture(&offset, &k_fsi::texSolventParticles, solvent, &k_fsi::texSolventParticles.channelDesc, sizeof(float) * 6 * n));

    nc = XS * YS * ZS;
    CC(cudaBindTexture(&offset, &k_fsi::texCellsStart, cellsstart, &k_fsi::texCellsStart.channelDesc, sizeof(int) * nc));
    CC(cudaBindTexture(&offset, &k_fsi::texCellsCount, cellscount, &k_fsi::texCellsCount.channelDesc, sizeof(int) * nc));
                       
}
}

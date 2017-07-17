namespace fsi {
void setup(const Particle *const solvent, const int npsolvent,
           const int *const cellsstart, const int *const cellscount) {
    if (firsttime) {
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
        firsttime = false;
    }

    size_t textureoffset = 0;

    if (npsolvent) {
        CC(cudaBindTexture(&textureoffset, &k_fsi::texSolventParticles, solvent,
                           &k_fsi::texSolventParticles.channelDesc,
                           sizeof(float) * 6 * npsolvent));
    }

    const int ncells = XS * YS * ZS;

    CC(cudaBindTexture(&textureoffset, &k_fsi::texCellsStart, cellsstart,
                       &k_fsi::texCellsStart.channelDesc, sizeof(int) * ncells));

    CC(cudaBindTexture(&textureoffset, &k_fsi::texCellsCount, cellscount,
                       &k_fsi::texCellsCount.channelDesc, sizeof(int) * ncells));
}
}

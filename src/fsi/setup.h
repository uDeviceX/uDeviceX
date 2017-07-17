namespace k_fsi {
void setup(const Particle *const solvent, const int npsolvent,
           const int *const cellsstart, const int *const cellscount) {
    if (firsttime) {
        texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
        texCellsStart.filterMode = cudaFilterModePoint;
        texCellsStart.mipmapFilterMode = cudaFilterModePoint;
        texCellsStart.normalized = 0;

        texCellsCount.channelDesc = cudaCreateChannelDesc<int>();
        texCellsCount.filterMode = cudaFilterModePoint;
        texCellsCount.mipmapFilterMode = cudaFilterModePoint;
        texCellsCount.normalized = 0;

        texSolventParticles.channelDesc = cudaCreateChannelDesc<float2>();
        texSolventParticles.filterMode = cudaFilterModePoint;
        texSolventParticles.mipmapFilterMode = cudaFilterModePoint;
        texSolventParticles.normalized = 0;

        CC(cudaFuncSetCacheConfig(interactions_3tpp, cudaFuncCachePreferL1));

        firsttime = false;
    }

    size_t textureoffset = 0;

    if (npsolvent) {
        CC(cudaBindTexture(&textureoffset, &texSolventParticles, solvent,
                           &texSolventParticles.channelDesc,
                           sizeof(float) * 6 * npsolvent));
    }

    const int ncells = XS * YS * ZS;

    CC(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart,
                       &texCellsStart.channelDesc, sizeof(int) * ncells));

    CC(cudaBindTexture(&textureoffset, &texCellsCount, cellscount,
                       &texCellsCount.channelDesc, sizeof(int) * ncells));
}
}

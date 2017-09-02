namespace lforces {
inline void setup_cloud(/* dummy c */ ) {
    /* call once */
    texParticlesF4.channelDesc = cudaCreateChannelDesc<float4>();
    texParticlesF4.filterMode = cudaFilterModePoint;
    texParticlesF4.mipmapFilterMode = cudaFilterModePoint;
    texParticlesF4.normalized = 0;

    texParticlesH4.channelDesc = cudaCreateChannelDescHalf4();
    texParticlesH4.filterMode = cudaFilterModePoint;
    texParticlesH4.mipmapFilterMode = cudaFilterModePoint;
    texParticlesH4.normalized = 0;
};

inline void ini_cloud(float4 *zip0, ushort4 *zip1, int n /* dummy c */ ) {
    size_t offset;
    CC(cudaBindTexture(&offset, &texParticlesF4, zip0, &texParticlesF4.channelDesc, sizeof(float)*8*n));
    CC(cudaBindTexture(&offset, &texParticlesH4, zip1, &texParticlesH4.channelDesc, sizeof(ushort4)*n));
};
} /* namespace */

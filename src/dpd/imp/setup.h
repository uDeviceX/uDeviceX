static setupt() {
    texStartAndCount.channelDesc = cudaCreateChannelDesc<uint2>();
    texStartAndCount.filterMode  = cudaFilterModePoint;
    texStartAndCount.mipmapFilterMode = cudaFilterModePoint;
    texStartAndCount.normalized = 0;

    texParticlesF4.channelDesc = cudaCreateChannelDesc<float4>();
    texParticlesF4.filterMode = cudaFilterModePoint;
    texParticlesF4.mipmapFilterMode = cudaFilterModePoint;
    texParticlesF4.normalized = 0;

    texParticlesH4.channelDesc = cudaCreateChannelDescHalf4();
    texParticlesH4.filterMode = cudaFilterModePoint;
    texParticlesH4.mipmapFilterMode = cudaFilterModePoint;
    texParticlesH4.normalized = 0;
}

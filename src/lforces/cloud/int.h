namespace lforces {
inline void ini_cloud(float4 *zip0, ushort4 *zip1, int n /* dummy c */ ) {
    size_t offset;
    CC(cudaBindTexture(&offset, &texParticlesF4, zip0, &texParticlesF4.channelDesc, sizeof(float)*8*n));
    CC(cudaBindTexture(&offset, &texParticlesH4, zip1, &texParticlesH4.channelDesc, sizeof(ushort4)*n));
};
} /* namespace */

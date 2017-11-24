static void setup() {
    texStartAndCount.channelDesc = cudaCreateChannelDesc<uint2>();
    texStartAndCount.filterMode  = cudaFilterModePoint;
    texStartAndCount.mipmapFilterMode = cudaFilterModePoint;
    texStartAndCount.normalized = 0;
}

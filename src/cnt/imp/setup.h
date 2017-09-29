static void setup() {
    dev::c::starts.channelDesc = cudaCreateChannelDesc<int>();
    dev::c::starts.filterMode = cudaFilterModePoint;
    dev::c::starts.mipmapFilterMode = cudaFilterModePoint;
    dev::c::starts.normalized = 0;

    dev::c::id.channelDesc = cudaCreateChannelDesc<int>();
    dev::c::id.filterMode = cudaFilterModePoint;
    dev::c::id.mipmapFilterMode = cudaFilterModePoint;
    dev::c::id.normalized = 0;
}

static void setup() {
    dev::c::start.channelDesc = cudaCreateChannelDesc<int>();
    dev::c::start.filterMode = cudaFilterModePoint;
    dev::c::start.mipmapFilterMode = cudaFilterModePoint;
    dev::c::start.normalized = 0;

    dev::c::id.channelDesc = cudaCreateChannelDesc<int>();
    dev::c::id.filterMode = cudaFilterModePoint;
    dev::c::id.mipmapFilterMode = cudaFilterModePoint;
    dev::c::id.normalized = 0;
}

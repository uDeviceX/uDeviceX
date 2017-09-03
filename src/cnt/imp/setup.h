void setup() {
    dev::t::start.channelDesc = cudaCreateChannelDesc<int>();
    dev::t::start.filterMode = cudaFilterModePoint;
    dev::t::start.mipmapFilterMode = cudaFilterModePoint;
    dev::t::start.normalized = 0;

    dev::t::id.channelDesc = cudaCreateChannelDesc<int>();
    dev::t::id.filterMode = cudaFilterModePoint;
    dev::t::id.mipmapFilterMode = cudaFilterModePoint;
    dev::t::id.normalized = 0;
}

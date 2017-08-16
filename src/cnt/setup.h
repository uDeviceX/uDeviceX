namespace cnt {
void setup() {
    k_cnt::t::start.channelDesc = cudaCreateChannelDesc<int>();
    k_cnt::t::start.filterMode = cudaFilterModePoint;
    k_cnt::t::start.mipmapFilterMode = cudaFilterModePoint;
    k_cnt::t::start.normalized = 0;

    k_cnt::t::id.channelDesc = cudaCreateChannelDesc<int>();
    k_cnt::t::id.filterMode = cudaFilterModePoint;
    k_cnt::t::id.mipmapFilterMode = cudaFilterModePoint;
    k_cnt::t::id.normalized = 0;
}
}

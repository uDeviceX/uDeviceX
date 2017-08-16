namespace fsi {
static void setup_first() {
    k_fsi::t::start.channelDesc = cudaCreateChannelDesc<int>();
    k_fsi::t::start.filterMode = cudaFilterModePoint;
    k_fsi::t::start.mipmapFilterMode = cudaFilterModePoint;
    k_fsi::t::start.normalized = 0;

    k_fsi::t::count.channelDesc = cudaCreateChannelDesc<int>();
    k_fsi::t::count.filterMode = cudaFilterModePoint;
    k_fsi::t::count.mipmapFilterMode = cudaFilterModePoint;
    k_fsi::t::count.normalized = 0;

    k_fsi::t::pp.channelDesc = cudaCreateChannelDesc<float2>();
    k_fsi::t::pp.filterMode = cudaFilterModePoint;
    k_fsi::t::pp.mipmapFilterMode = cudaFilterModePoint;
    k_fsi::t::pp.normalized = 0;
}

void setup(const Particle *const pp, int n, const int *const cellsstart, const int *const cellscount) {
    size_t offset;
    int nc;
    if (firsttime) {
        setup_first();
        firsttime = false;
    }

    offset = 0;
    if (n)
        CC(cudaBindTexture(&offset, &k_fsi::t::pp, pp, &k_fsi::t::pp.channelDesc, sizeof(float) * 6 * n));

    nc = XS * YS * ZS;
    CC(cudaBindTexture(&offset, &k_fsi::t::start, cellsstart, &k_fsi::t::start.channelDesc, sizeof(int) * nc));
    CC(cudaBindTexture(&offset, &k_fsi::t::count, cellscount, &k_fsi::t::count.channelDesc, sizeof(int) * nc));
                       
}
}

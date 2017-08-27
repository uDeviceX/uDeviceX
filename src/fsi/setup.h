namespace fsi {
static void setup_first() {
    dev::t::start.channelDesc = cudaCreateChannelDesc<int>();
    dev::t::start.filterMode = cudaFilterModePoint;
    dev::t::start.mipmapFilterMode = cudaFilterModePoint;
    dev::t::start.normalized = 0;

    dev::t::pp.channelDesc = cudaCreateChannelDesc<float2>();
    dev::t::pp.filterMode = cudaFilterModePoint;
    dev::t::pp.mipmapFilterMode = cudaFilterModePoint;
    dev::t::pp.normalized = 0;
}

void setup(const Particle *const pp, int n, const int *const cellsstart) {
    size_t offset;
    int nc;
    if (firsttime) {
        setup_first();
        firsttime = false;
    }

    offset = 0;
    if (n)
        CC(cudaBindTexture(&offset, &dev::t::pp, pp, &dev::t::pp.channelDesc, sizeof(float) * 6 * n));

    nc = XS * YS * ZS;
    CC(cudaBindTexture(&offset, &dev::t::start, cellsstart, &dev::t::start.channelDesc, sizeof(int) * nc));
                       
}
}

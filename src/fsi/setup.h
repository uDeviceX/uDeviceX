namespace fsi {
static void setup_first() {
    dev::t::start.channelDesc = cudaCreateChannelDesc<int>();
    dev::t::start.filterMode = cudaFilterModePoint;
    dev::t::start.mipmapFilterMode = cudaFilterModePoint;
    dev::t::start.normalized = 0;
}

void setup(const Particle *const pp, int n, const int *const cellsstart) {
    size_t offset;
    int nc;
    if (firsttime) {
        setup_first();
        firsttime = false;
    }

    offset = 0;
    nc = XS * YS * ZS;
    CC(cudaBindTexture(&offset, &dev::t::start, cellsstart, &dev::t::start.channelDesc, sizeof(int) * nc));
}
}

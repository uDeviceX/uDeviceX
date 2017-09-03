void ini() {
    Dalloc(&g::starts, sz);
    Dalloc(&g::counts, sz);
    scan::alloc_work(sz, &ws);
    entries = new DeviceBuffer<int>;
    indexes = new DeviceBuffer<uchar4>;
    rgen = new rnd::KISS;
    *rgen = rnd::KISS(7119 - m::rank, 187 + m::rank, 18278, 15674);
}

namespace cnt {
void ini() {
    Dalloc0(&starts, sz);
    Dalloc0(&counts, sz);
    scan::alloc_work(sz, &ws);
    entries = new DeviceBuffer<int>;
    indexes = new DeviceBuffer<uchar4>;
    rgen = new rnd::KISS;
    *rgen = rnd::KISS(7119 - m::rank, 187 + m::rank, 18278, 15674);
}
}

static void ini_rnd() {
    g::rgen = new rnd::KISS;
    *g::rgen =
        rnd::KISS(7119 - m::rank,
                  187 + m::rank, 18278, 15674);
}

void ini() {
    ini_rnd();
    Dalloc(&g::starts, sz);
    Dalloc(&g::counts, sz);
    scan::alloc_work(sz, &g::ws);
    entries = new DeviceBuffer<int>;
    indexes = new DeviceBuffer<uchar4>;
}

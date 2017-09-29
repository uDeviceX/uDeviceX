static void ini_rnd() {
    g::rgen = new rnd::KISS;
    *g::rgen =
        rnd::KISS(7119 - m::rank,
                  187 + m::rank, 18278, 15674);
}

void ini() {
    ini_rnd();
    Dalloc(&g::starts, g::sz);
    Dalloc(&g::counts, g::sz);
    scan::alloc_work(g::sz, &g::ws);
    g::entries = new DeviceBuffer<int>;
    g::indexes = new DeviceBuffer<uchar4>;
}

namespace cnt {
void ini() {
    Palloc(&starts, XS*YS*ZS + 16);
    counts = new DeviceBuffer<int>(XS*YS*ZS + 16);
    scan::alloc_work(XS*YS*ZS + 16, &ws);
    entries = new DeviceBuffer<int>;
    indexes = new DeviceBuffer<uchar4>;
    rgen = new rnd::KISS;
    *rgen = rnd::KISS(7119 - m::rank, 187 + m::rank, 18278, 15674);
}
}

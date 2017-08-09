namespace cnt {
void ini() {
    cellsstart = new DeviceBuffer<int>(XS*YS*ZS + 16);
    cellscount = new DeviceBuffer<int>(XS*YS*ZS + 16);
    scan::alloc_work(XS*YS*ZS + 16, &ws);
    cellsentries = new DeviceBuffer<int>;
    subindices = new DeviceBuffer<uchar4>;
    rgen = new rnd::KISS;
    *rgen = rnd::KISS(7119 - m::rank, 187 + m::rank, 18278, 15674);
}
}

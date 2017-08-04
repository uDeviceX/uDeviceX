namespace fsi {
void ini() {
    local_trunk = new rnd::KISS;
    wsolvent    = new SolventWrap;
    *local_trunk = rnd::KISS(1908 - m::rank, 1409 + m::rank, 290, 12968);
}
}

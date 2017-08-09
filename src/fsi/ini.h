namespace fsi {
void ini() {
    rgen = new rnd::KISS;
    wsolvent    = new SolventWrap;
    *rgen = rnd::KISS(1908 - m::rank, 1409 + m::rank, 290, 12968);
}
}

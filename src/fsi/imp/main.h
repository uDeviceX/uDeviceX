void ini(Fsi *fsi) {
    fsi->rgen = new rnd::KISS(1908 - m::rank, 1409 + m::rank, 290, 12968);
    fsi->wo   = new SolventWrap;
}

void fin(Fsi *fsi) {
    delete fsi->rgen;
    delete fsi->wo;
}

void bind(SolventWrap wrap, /**/ Fsi *fsi) {
    *(fsi->wo) = wrap;
}

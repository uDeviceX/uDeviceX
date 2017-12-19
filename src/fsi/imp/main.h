void ini(Fsi *fsi) {
    UC(rnd_ini(1908 - m::rank, 1409 + m::rank, 290, 12968, /**/ &fsi->rgen));
    fsi->wo   = new SolventWrap;
}

void fin(Fsi *fsi) {
    UC(rnd_fin(fsi->rgen));
    delete fsi->wo;
}

void bind(SolventWrap wrap, /**/ Fsi *fsi) {
    *(fsi->wo) = wrap;
}

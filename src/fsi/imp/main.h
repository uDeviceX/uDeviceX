void fsi_ini(int rank, /**/ Fsi *fsi) {
    UC(rnd_ini(1908 - rank, 1409 + rank, 290, 12968, /**/ &fsi->rgen));
    fsi->wo   = new SolventWrap;
}

void fsi_fin(Fsi *fsi) {
    UC(rnd_fin(fsi->rgen));
    delete fsi->wo;
}

void fsi_bind_solvent(Cloud c, Force *ff, int n, int *starts, /**/ Fsi *fsi) {
    SolventWrap *w = fsi->wo;
    w->c = c;
    w->n = n;
    w->ff = ff;
    w->starts = starts;
}


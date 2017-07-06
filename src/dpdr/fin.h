void fin_tcom(/**/ MPI_Comm *cart) {
    MC(l::m::Comm_free(cart));
}

void fin_trnd(/**/ l::rnd::d::KISS* interrank_trunks[]) {
    for (int i = 0; i < 26; ++i) delete interrank_trunks[i];
}

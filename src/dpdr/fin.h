void fin_tcom(const bool first, /**/ MPI_Comm *cart, Reqs *sreq, Reqs *rreq) {
    if (!first) {
        wait_req(sreq);
        cancel_req(rreq);
    }

    MC(l::m::Comm_free(cart));
}

void fin_trnd(/**/ l::rnd::d::KISS* interrank_trunks[]) {
    for (int i = 0; i < 26; ++i) delete interrank_trunks[i];
}

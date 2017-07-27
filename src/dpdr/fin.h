namespace dpdr {
namespace sub {

static void cancel_req(MPI_Request r[26]) {
    for (int i = 0; i < 26; ++i) MC(MPI_Cancel(r + i));
}
    
static void cancel_Reqs(Reqs *r) {
    cancel_req(r->pp);
    cancel_req(r->cells);
    cancel_req(r->counts);
}

void fin_tcom(const bool first, /**/ MPI_Comm *cart, Reqs *sreq, Reqs *rreq) {
    if (!first) {
        wait_Reqs(sreq);
        cancel_Reqs(rreq);
    }

    MC(l::m::Comm_free(cart));
}

void fin_trnd(/**/ rnd::KISS* interrank_trunks[]) {
    for (int i = 0; i < 26; ++i) delete interrank_trunks[i];
}


} // sub
} // dpdr

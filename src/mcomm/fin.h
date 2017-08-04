namespace mcomm {
namespace sub {

void fin_tcom(const bool first, /**/ MPI_Comm *cart, Reqs *sreq, Reqs *rreq) {
    if (!first) {
        wait_req(sreq);
        cancel_req(rreq);
    }

    MC(l::m::Comm_free(cart));
}

} // sub
} // mcomm

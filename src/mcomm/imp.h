/* Structure containing all kinds of requests a fragment can have */
struct Reqs {
    MPI_Request pp[26], counts[26];
};

void cancel_req(Reqs *r) {
    for (int i = 0; i < 26; ++i) {
        MC(MPI_Cancel(r->pp     + i));
        MC(MPI_Cancel(r->counts + i));
    }
}

void wait_req(Reqs *r) {
    MPI_Status ss[26];
    MC(l::m::Waitall(26, r->pp,     ss));
    MC(l::m::Waitall(26, r->counts, ss));
}

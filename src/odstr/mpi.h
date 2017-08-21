namespace odstr {
namespace sub {

int lsend(const void *buf, int count, MPI_Datatype datatype, int dest,
           int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int lrecv(void *buf, int count, MPI_Datatype datatype, int source,
          int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}


void waitall(MPI_Request *reqs) {
    MPI_Status statuses[123];
    l::m::Waitall(26, reqs, statuses);
}

void waitall_s(MPI_Request *reqs) {
    waitall(reqs);
}

void waitall_r(MPI_Request *reqs) {
    waitall(reqs) ;
}

void post_recv(const MPI_Comm cart, const int rank[], const int btc, const int btp,
               MPI_Request *size_req, MPI_Request *mesg_req, Recv *r) {
    void *buf;
    int count, source, tag;
    MPI_Request *request;
    int i, c;

    for(i = 1, c = 0; i < 27; ++i, ++c) {
        buf = r->size + i;
        count = 1;
        source = rank[i];
        tag = btc + r->tags[i];
        request = &size_req[c];
        lrecv(buf, count, MPI_INTEGER, source, tag, cart, request);
    }

    for(i = 1, c = 0; i < 27; ++i, ++c) {
        buf = r->pp.hst[i];
        count = MAX_PART_NUM;
        source = rank[i];
        tag = btp + r->tags[i];
        request = &mesg_req[c];
        lrecv(buf, count, datatype::particle, source, tag, cart, request);
    }
}

void send_sz(MPI_Comm cart, const int rank[], const int bt, /**/ Send *s, MPI_Request *req) {
    const void *buf;
    int count, dest, tag;
    MPI_Request *request;
    int i, c;

    for(i = 1, c = 0; i < 27; ++i, ++c) {
        buf = s->size + i;
        count = 1;
        dest = rank[i];
        tag = bt + i;
        request = &req[c];
        lsend(buf, count, MPI_INTEGER, dest, tag, cart, request);
    }
}

void send_pp(MPI_Comm cart, const int rank[], const int bt, /**/ Send *s, MPI_Request *req) {
    const void *buf;
    int count, dest, tag;
    MPI_Request *request;
    int i, c;
    for(i = 1, c = 0; i < 27; ++i, ++c) {
        buf = s->pp.hst[i];
        count = s->size[i];
        dest = rank[i];
        tag = bt + i;
        request = &req[c];
        lsend(buf, count, datatype::particle, dest, tag, cart, request);
    }
}

/* TODO: this is not used, why? */
void cancel_recv(/**/ MPI_Request *size_req, MPI_Request *mesg_req) {
    for(int i = 0; i < 26; ++i) l::m::Cancel(size_req + i) ;
    for(int i = 0; i < 26; ++i) l::m::Cancel(mesg_req + i) ;
}

} /* namespace */
} /* namespace */

namespace odstr {
namespace sub {

void send_ii(const int rank[], const int size[], const int bt, /**/ Pbufs<int> *sii, MPI_Request *req) {
    for(int i = 1, cnt = 0; i < 27; ++i)
    m::Isend(sii->hst[i], size[i], MPI_INT, rank[i],
                bt + i, m::cart, &req[cnt++]);
}

void post_recv_ii(const int rank[], const int tags[], const int bt, /**/ MPI_Request *ii_req, Pbufs<int> *rii) {
    for(int i = 1, c = 0; i < 27; ++i)
    m::Irecv(rii->hst[i], MAX_PART_NUM, MPI_INT, rank[i],
                bt + tags[i], m::cart, ii_req + c++);
}

}
}

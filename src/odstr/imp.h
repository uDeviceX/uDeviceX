void Distr::waitall(MPI_Request *reqs) {
    MPI_Status statuses[128]; /* big number */
    l::m::Waitall(26, reqs, statuses) ;
}

void Distr::post_recv(MPI_Comm cart, int rank[],
                      MPI_Request *size_req, MPI_Request *mesg_req) {
    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(r.size + i, 1, MPI_INTEGER, rank[i],
                BT_C_ODSTR + r.tags[i], cart, size_req + c++);

    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(r.pp_hst[i], MAX_PART_NUM, MPI_FLOAT, rank[i],
                BT_P_ODSTR + r.tags[i], cart, mesg_req + c++);
}

void Distr::post_recv_ii(MPI_Comm cart, int rank[],
                         MPI_Request *ii_req) {
    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(r.ii_hst[i], MAX_PART_NUM, MPI_INT, rank[i],
                BT_I_ODSTR + r.tags[i], cart, ii_req + c++);
}

void Distr::halo(Particle *pp, int n) {
    CC(cudaMemset(s.size_dev, 0,  27*sizeof(s.size_dev[0])));
    dev::halo<<<k_cnf(n)>>>(pp, n, /**/ s.iidx, s.size_dev);
}

void Distr::scan(int n) {
    dev::scan<<<1, 32>>>(n, s.size_dev, /**/ s.strt, s.size_pin->DP);
    dSync();
}

void Distr::pack_pp(const Particle *pp, int n) {
    dev::pack<float2, 3> <<<k_cnf(3*n)>>>((float2*)pp, s.iidx, s.strt, /**/ s.pp_dev);
}

void Distr::pack_ii(const int *ii, int n) {
    dev::pack<int, 1> <<<k_cnf(n)>>>(ii, s.iidx, s.strt, /**/ s.ii_dev);
}

int Distr::send_sz(MPI_Comm cart, int rank[], MPI_Request *req) {
    for(int i = 0; i < 27; ++i) s.size[i] = s.size_pin->D[i];
    for(int i = 1, cnt = 0; i < 27; ++i)
    l::m::Isend(s.size + i, 1, MPI_INTEGER, rank[i],
                BT_C_ODSTR + i, cart, &req[cnt++]);
    return s.size[0]; /* `n' bulk */
}

void Distr::send_msg(MPI_Comm cart, int rank[], MPI_Request *req) {
    for(int i = 1, cnt = 0; i < 27; ++i)
    l::m::Isend(s.pp_hst[i], s.size[i] * 6, MPI_FLOAT, rank[i],
                BT_P_ODSTR + i, cart, &req[cnt++]);
}

void Distr::recv_count(int *nhalo) {
    int i;
    static int size[27], strt[28];

    size[0] = strt[0] = 0;
    for (i = 1; i < 27; ++i)    size[i] = r.size[i];
    for (i = 1; i < 28; ++i)    strt[i] = strt[i - 1] + size[i - 1];
    CC(cudaMemcpy(r.strt,    strt,    sizeof(strt),    H2D));
    *nhalo = strt[27];
}

void Distr::unpack_pp(int n, /*o*/ Particle *pp_re) {
    dev::unpack<float2, 3> <<<k_cnf(3*n)>>> (r.pp_dev, r.strt, /**/ (float2*) pp_re);
}

void Distr::unpack_ii(int n, /*o*/ int *ii_re) {
    dev::unpack<int, 1> <<<k_cnf(n)>>> (r.ii_dev, r.strt, /**/ ii_re);
}

void Distr::subindex_remote(int n, /*io*/ Particle *pp_re, int *counts, /**/ uchar4 *subi) {
    dev::subindex_remote <<<k_cnf(n)>>> (n, r.strt, /*io*/ (float2*) pp_re, counts, /**/ subi);
}

void Distr::cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req) {
    for(int i = 0; i < 26; ++i) l::m::Cancel(size_req + i) ;
    for(int i = 0; i < 26; ++i) l::m::Cancel(mesg_req + i) ;
}

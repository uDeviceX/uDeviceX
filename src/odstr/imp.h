void Fluid::waitall(MPI_Request *reqs) {
    MPI_Status statuses[128]; /* big number */
    l::m::Waitall(26, reqs, statuses) ;
}

void Fluid::post_recv(MPI_Comm cart, int rank[],
                      MPI_Request *size_req, MPI_Request *mesg_req) {
    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(r.size + i, 1, MPI_INTEGER, rank[i],
                950 + r.tags[i], cart, size_req + c++);

    for(int i = 1, c = 0; i < 27; ++i)
    l::m::Irecv(r.hst[i], MAX_PART_NUM, MPI_FLOAT, rank[i],
                950 + r.tags[i] + 333, cart, mesg_req + c++);
}

void Fluid::halo(Particle *pp, int n) {
    CC(cudaMemset(s.size_dev, 0,  27*sizeof(s.size_dev[0])));
    dev::halo<<<k_cnf(n)>>>(pp, n, /**/ s.iidx, s.size_dev);
}

void Fluid::scan(int n) {
    dev::scan<<<1, 32>>>(n, s.size_dev, /**/ s.strt, s.size_pin->DP);
    dSync();
}

void Fluid::pack(Particle *pp, int n) {
    dev::pack<<<k_cnf(3*n)>>>((float2*)pp, s.iidx, s.strt, /**/ s.dev);
    dSync();
}

int Fluid::send_sz(MPI_Comm cart, int rank[], MPI_Request *req) {
    for(int i = 0; i < 27; ++i) s.size[i] = s.size_pin->D[i];
    for(int i = 1, cnt = 0; i < 27; ++i)
    l::m::Isend(s.size + i, 1, MPI_INTEGER, rank[i],
                950 + i, cart, &req[cnt++]);
    return s.size[0]; /* `n' bulk */
}

void Fluid::send_msg(MPI_Comm cart, int rank[], MPI_Request *req) {
    for(int i = 1, cnt = 0; i < 27; ++i)
    l::m::Isend(s.hst[i], s.size[i] * 6, MPI_FLOAT, rank[i],
                950 + i + 333, cart, &req[cnt++]);
}

void Fluid::recv_count(int *nhalo_padded, int *nhalo) {
    int i;
    static int size[27], strt[28], strt_pa[28];

    size[0] = strt[0] = strt_pa[0] = 0;
    for (i = 1; i < 27; ++i)    size[i] = r.size[i];
    for (i = 1; i < 28; ++i)    strt[i] = strt[i - 1] + size[i - 1];
    for (i = 1; i < 28; ++i) strt_pa[i] = strt_pa[i - 1] + 32 * ceiln(size[i-1], 32);
    CC(cudaMemcpy(r.strt,    strt,    sizeof(strt),    H2D));
    CC(cudaMemcpy(r.strt_pa, strt_pa, sizeof(strt_pa), H2D));
    *nhalo = strt[27];
    *nhalo_padded = strt_pa[27];
}

void Fluid::unpack(int n_pa,
                   /*io*/ int *count,
                   /*o*/ uchar4 *subi, Particle *pp_re) {
    /* n_pa: n padded */
    dev::unpack<<<k_cnf(n_pa)>>>
        (n_pa,  r.dev, r.strt, r.strt_pa,
         /*io*/ count,
         /*o*/ (float2*)pp_re, subi);
}

void Fluid::cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req) {
    for(int i = 0; i < 26; ++i) l::m::Cancel(size_req + i) ;
    for(int i = 0; i < 26; ++i) l::m::Cancel(mesg_req + i) ;
}

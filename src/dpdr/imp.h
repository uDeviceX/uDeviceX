using dev::int26;
using dev::int27;
using dev::intp26;
using dev::Particlep26;

/* Structure containing all kinds of requests a fragment can have */
struct Reqs {
    MPI_Request pp[26], cells[26], counts[26];
};

void cancel_req(MPI_Request r[26]) {
    for (int i = 0; i < 26; ++i) MC(MPI_Cancel(r + i));
}
    
void cancel_Reqs(Reqs *r) {
    cancel_req(r->pp);
    cancel_req(r->cells);
    cancel_req(r->counts);
}

void wait_req(MPI_Request r[26]) {
    MPI_Status ss[26];
    MC(l::m::Waitall(26, r, ss));
}

void wait_Reqs(Reqs *r) {
    wait_req(r->cells);
    wait_req(r->pp);
    wait_req(r->counts);
}

void gather_cells(const int *start, const int *count, const int27 fragstarts, const int26 fragnc,
                  const int ncells, /**/ intp26 fragstr, intp26 fragcnt, intp26 fragcum) {
    if (ncells) dev::count<<<k_cnf(ncells)>>>(fragstarts, start, count, fragstr, fragcnt);
    dev::scan<32><<<26, 32 * 32>>>(fragnc, fragcnt, /**/ fragcum);
}

void copy_cells(const int27 fragstarts, const int ncells, const intp26 srccells, /**/ intp26 dstcells) {
    if (ncells) dev::copycells<<<k_cnf(ncells)>>>(fragstarts, srccells, /**/ dstcells);
}
  
void pack(const int27 fragstarts, const int ncells, const Particle *pp, const intp26 fragstr,
          const intp26 fragcnt, const intp26 fragcum, const int26 fragcapacity, /**/ intp26 fragii, Particlep26 fragpp, int *bagcounts) {
    if (ncells)
        dev::fill_all<<<(ncells + 1) / 2, 32>>>(fragstarts, pp, fragstr, fragcnt, fragcum,
                                                fragcapacity, /**/ fragii, fragpp, bagcounts);
}

void pack_ii(const int27 fragstarts, const int ncells, const int *ii, const intp26 fragstr, const intp26 fragcnt, const intp26 fragcum,
             const int26 fragcapacity, /**/ intp26 fragii) {
    if (ncells)
        dev::fill_all_ii<<<(ncells + 1) / 2, 32>>>(fragstarts, ii, fragstr, fragcnt, fragcum, fragcapacity, fragii);
}

void copy_pp(const int *fragnp, const Particlep26 fragppdev, /**/ Particlep26 fragpphst) {
    //dSync(); /* wait for fill_all */ /* use async copy now, no need to wait */
    
    for (int i = 0; i < 26; ++i)
        if (fragnp[i])
            CC(cudaMemcpyAsync(fragpphst.d[i], fragppdev.d[i], sizeof(Particle) * fragnp[i], D2H));
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */
}

void copy_ii(const int *fragnp, const intp26 fragiidev, /**/ intp26 fragiihst) {
    dSync(); /* wait for fill_all_ii */
    
    for (int i = 0; i < 26; ++i)
        if (fragnp[i])
            CC(cudaMemcpyAsync(fragiihst.d[i], fragiidev.d[i], sizeof(int) * fragnp[i], D2H));
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */
}

void post_send(MPI_Comm cart, const int dstranks[], const int *fragnp, const int26 fragnc, const intp26 fragcum,
               const Particlep26 fragpp, int btcs, int btc, int btp, /**/ Reqs *sreq) {

    for (int i = 0; i < 26; ++i) {
        const int nc = fragnc.d[i];
        MC(l::m::Isend(fragcum.d[i], nc, MPI_INT, dstranks[i],
                       btcs + i, cart, sreq->cells + i));

        const int count = fragnp[i];
        
        MC(l::m::Isend(&count, 1, MPI_INT, dstranks[i],
                       btc + i, cart, sreq->counts + i));
        
        MC(l::m::Isend(fragpp.d[i], count, datatype::particle,
                       dstranks[i], btp + i, cart, sreq->pp + i));
    }
}

void post_send_ii(MPI_Comm cart, const int dstranks[], const int *fragnp,
                  const intp26 fragii, int bt, /**/ MPI_Request sreq[26]) {

    for (int i = 0; i < 26; ++i)
        MC(l::m::Isend(fragii.d[i], fragnp[i], MPI_INT, dstranks[i], bt + i, cart, sreq + i));
}

void post_expected_recv(MPI_Comm cart, const int dstranks[], const int recv_tags[], const int estimate[], const int26 fragnc,
                        int btcs, int btc, int btp, /**/ Particlep26 fragpp, int *Rfragnp, intp26 Rfragcum, Reqs *rreq) {
    for (int i = 0; i < 26; ++i) {
        MC(l::m::Irecv(fragpp.d[i], estimate[i], datatype::particle, dstranks[i], btp + recv_tags[i],
                       cart, rreq->pp + i));
    
        MC(l::m::Irecv(Rfragcum.d[i], fragnc.d[i], MPI_INT, dstranks[i],
                       btcs + recv_tags[i], cart, rreq->cells + i));
    
        MC(l::m::Irecv(Rfragnp + i, 1, MPI_INT, dstranks[i],
                       btc + recv_tags[i], cart, rreq->counts + i));
    }
}

void post_expected_recv_ii(MPI_Comm cart, const int dstranks[], const int recv_tags[], const int estimate[],
                           int bt, /**/ intp26 fragii, MPI_Request rreq[26]) {
    for (int i = 0; i < 26; ++i)
        MC(l::m::Irecv(fragii.d[i], estimate[i], MPI_INT, dstranks[i], bt + recv_tags[i], cart, rreq + i));
}

void recv(const int *np, const int *nc, /**/ Rbufs *b) {
    for (int i = 0; i < 26; ++i)
        if (np[i] > 0) CC(cudaMemcpyAsync(b->pp.d[i], b->ppdev.d[i], sizeof(Particle) * np[i], D2D));

    for (int i = 0; i < 26; ++i)
        if (nc[i] > 0) CC(cudaMemcpyAsync(b->cum.d[i], b->cumdev.d[i],  sizeof(int) * nc[i], D2D));
}

void recv_ii(const int *np, /**/ RIbuf *b) {
    for (int i = 0; i < 26; ++i)
        if (np[i] > 0) CC(cudaMemcpyAsync(b->ii.d[i], b->iidev.d[i], sizeof(int) * np[i], D2D));
}

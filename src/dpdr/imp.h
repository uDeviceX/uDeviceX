typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

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
    dev::fill_all<<<(ncells + 1) / 2, 32>>>(fragstarts, pp, bagcounts, fragstr, fragcnt, fragcum,
                                            fragcapacity, fragii, fragpp);
}

void copy_pp(const int *fragnp, const Particlep26 fragppdev, /**/ Particlep26 fragpphst) {
    dSync(); /* wait for fill_all */
    
    for (int i = 0; i < 26; ++i)
    if (fragnp[i])
    cudaMemcpyAsync(fragpphst.d[i], fragppdev.d[i], sizeof(Particle) * fragnp[i], D2H);
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */
}

void post(MPI_Comm cart, const int dstranks[], const int *fragnp, const int26 fragnc, const intp26 fragcum, const Particlep26 fragpp,
          /**/ MPI_Request *sendcellsreq, MPI_Request *sendcountreq, MPI_Request *sendreq) {

    for (int i = 0; i < 26; ++i) {
        const int nc = fragnc.d[i];
        MC(l::m::Isend(fragcum.d[i], nc, MPI_INT, dstranks[i],
                       BT_CS_DPD + i, cart, sendcellsreq + i));

        const int count = fragnp[i];
        
        MC(l::m::Isend(&count, 1, MPI_INT, dstranks[i],
                       BT_C_DPD + i, cart, sendcountreq + i));
        
        MC(l::m::Isend(fragpp.d[i], count, Particle::datatype(),
                       dstranks[i], BT_P_DPD + i, cart, sendreq + i));
    }
}

namespace cnt {
void build(std::vector<ParticlesWrap> wr) {
    /* build cells */
    no = wr.size();
    int ntotal = 0;
    for (int i = 0; i < (int) wr.size(); ++i) ntotal += wr[i].n;

    indexes->resize(ntotal);
    entries->resize(ntotal);

    CC(cudaMemsetAsync(counts->D, 0, sizeof(int) * counts->S));


    int ctr = 0;
    for (int i = 0; i < (int) wr.size(); ++i) {
        ParticlesWrap it = wr[i];
        KL(k_common::subindex_local<true>, (k_cnf(it.n)), (it.n, (float2 *)it.p, counts->D, indexes->D + ctr));
        ctr += it.n;
    }

    scan::scan(counts->D, XS*YS*ZS + 16, /**/ starts->D, /*w*/ &ws);

    ctr = 0;
    for (int i = 0; i < (int) wr.size(); ++i) {
        ParticlesWrap it = wr[i];
        KL(k_cnt::populate, (k_cnf(it.n)),
           (indexes->D + ctr, starts->D, it.n, i, ntotal, (k_cnt::CellEntry *)entries->D));
        ctr += it.n;
    }

    bind(starts->D, entries->D, ntotal, wr);
}
}

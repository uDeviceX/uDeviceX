void build(std::vector<ParticlesWrap> wr) {
    /* build cells */
    no = wr.size();
    int ntotal = 0;
    for (int i = 0; i < (int) wr.size(); ++i) ntotal += wr[i].n;

    indexes->resize(ntotal);
    entries->resize(ntotal);

    CC(cudaMemsetAsync(counts, 0, sizeof(*counts)*sz));


    int ctr = 0;
    for (int i = 0; i < (int) wr.size(); ++i) {
        ParticlesWrap it = wr[i];
        KL(k_index::local<true>, (k_cnf(it.n)), (it.n, (float2 *)it.p, counts, indexes->D + ctr));
        ctr += it.n;
    }

    scan::scan(counts, sz, /**/ starts, /*w*/ &ws);

    ctr = 0;
    for (int i = 0; i < (int) wr.size(); ++i) {
        ParticlesWrap it = wr[i];
        KL(dev::populate, (k_cnf(it.n)),
           (indexes->D + ctr, starts, it.n, i, ntotal, entries->D));
        ctr += it.n;
    }

    bind(starts, entries->D, ntotal, wr);
}

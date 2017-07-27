namespace cnt {
void build(std::vector<ParticlesWrap> wsolutes) {
    /* build cells */
    nsolutes = wsolutes.size();
    int ntotal = 0;
    for (int i = 0; i < (int) wsolutes.size(); ++i) ntotal += wsolutes[i].n;

    subindices->resize(ntotal);
    cellsentries->resize(ntotal);

    CC(cudaMemsetAsync(cellscount->D, 0, sizeof(int) * cellscount->S));


    int ctr = 0;
    for (int i = 0; i < (int) wsolutes.size(); ++i) {
        ParticlesWrap it = wsolutes[i];
        if (it.n)
        k_common::subindex_local<true><<<k_cnf(it.n)>>>
            (it.n, (float2 *)it.p, cellscount->D, subindices->D + ctr);
        ctr += it.n;
    }

    scan::scan(cellscount->D, k_cnt::NCELLS + 16, /**/ cellsstart->D, /*w*/ &ws);

    ctr = 0;
    for (int i = 0; i < (int) wsolutes.size(); ++i) {
        ParticlesWrap it = wsolutes[i];

        if (it.n)
        k_cnt::populate<<<k_cnf(it.n)>>>
            (subindices->D + ctr, cellsstart->D, it.n, i, ntotal,
             (k_cnt::CellEntry *)cellsentries->D);
        ctr += it.n;
    }

    bind(cellsstart->D, cellsentries->D, ntotal, wsolutes);
}
}

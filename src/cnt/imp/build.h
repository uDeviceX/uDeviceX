void bind(int nw, PaWrap *pw, FoWrap *fw) {
    /* build cells */
    int ntotal = 0;
    for (int i = 0; i < nw; ++i) ntotal += pw[i].n;

    indexes->resize(ntotal);
    entries->resize(ntotal);

    CC(cudaMemsetAsync(g::counts, 0, sizeof(*g::counts)*sz));

    int ctr = 0;
    for (int i = 0; i < nw; ++i) {
        PaWrap it = pw[i];
        KL(k_index::local<true>, (k_cnf(it.n)), (it.n, (float2 *)it.pp, g::counts, indexes->D + ctr));
        ctr += it.n;
    }

    scan::scan(g::counts, sz, /**/ g::starts, /*w*/ &ws);

    ctr = 0;
    for (int i = 0; i < nw; ++i) {
        PaWrap it = pw[i];
        KL(dev::populate, (k_cnf(it.n)),
           (indexes->D + ctr, g::starts, it.n, i, entries->D));
        ctr += it.n;
    }

    bind(g::starts, entries->D, ntotal, nw, pw, fw);
}

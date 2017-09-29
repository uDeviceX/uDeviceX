void bind(int nw, PaWrap *pw, FoWrap *fw) {
    /* build cells */
    int ntotal = 0;
    for (int i = 0; i < nw; ++i) ntotal += pw[i].n;

    g::indexes->resize(ntotal);
    g::entries->resize(ntotal);

    CC(cudaMemsetAsync(g::counts, 0, sizeof(*g::counts)*g::sz));

    int ctr = 0;
    for (int i = 0; i < nw; ++i) {
        PaWrap it = pw[i];
        KL(k_index::local<true>, (k_cnf(it.n)), (it.n, (float2 *)it.pp, g::counts, g::indexes->D + ctr));
        ctr += it.n;
    }

    scan::scan(g::counts, g::sz, /**/ g::starts, /*w*/ &g::ws);
    ctr = 0;
    for (int i = 0; i < nw; ++i) {
        PaWrap it = pw[i];
        KL(dev::populate, (k_cnf(it.n)),
           (g::indexes->D + ctr, g::starts, it.n, i, g::entries->D));
        ctr += it.n;
    }

    bind0(g::starts, g::entries->D, ntotal, nw, pw, fw);
}

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
}


void build_cells(int nw, const PaWrap *pw, /**/ Contact *c) {
    const PaWrap *w;
    int i, cc[MAX_OBJ_TYPES] = {0};

    clist::ini_counts(&c->cells);
    const bool project = true;

    for (i = 0; i < nw; ++i) {
        w = pw + i;
        cc[i] = w->n;
        clist::subindex(project, i, w->n, w->pp, /**/ &c->cells, &c->cmap);
    }
    clist::build_map(cc, /**/ &c->cells, &c->cmap);
}

// TODO: split remote and bulk subindex?

void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, /*w*/ Work *w) {
    build(nlo, 0, nout, pplo, NULL, /**/ ppout, c, /*w*/ w);
}


void build(int nlo, int nre, int nout, const Particle *pplo, const Particle *ppre, /**/ Particle *ppout, Clist *c, /*w*/ Work *w) {
    int nc, *cc, *ss;
    uchar4 *eelo, *eere;
    int3  dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;
    eelo = w->eelo;
    eere = w->eere;
    uint *ii = w->ii;
    
    CC(d::MemsetAsync(cc, 0, nc * sizeof(int)));

    if (nlo) KL(dev::subindex, (k_cnf(nlo)), (dims, nlo, pplo, /*io*/ cc, /**/ eelo));
    if (nre) KL(dev::subindex, (k_cnf(nre)), (dims, nre, ppre, /*io*/ cc, /**/ eere));

    scan::scan(cc, nc, /**/ ss, /*w*/ &w->scan);

    if (nlo) KL(dev::get_ids, (k_cnf(nlo)), (true,  dims, nlo, ss, eelo, /**/ ii));
    if (nre) KL(dev::get_ids, (k_cnf(nre)), (false, dims, nre, ss, eere, /**/ ii));

    KL(dev::gather, (k_cnf(nout)), (pplo, ppre, ii, nout, /**/ ppout));
}

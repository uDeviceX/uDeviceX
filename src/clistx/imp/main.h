// TODO: split remote and bulk subindex?

#define REMOTE (true)
#define LOCAL (false)

void ini_counts(Clist *c) {
    CC(d::MemsetAsync(c->counts, 0, c->ncells * sizeof(int)));
}

static void subindex(int n, const Particle *pp, int3 dims, /**/ int *cc, uchar4 *ee) {
    if (n) KL(dev::subindex, (k_cnf(n)), (dims, n, pp, /*io*/ cc, /**/ ee));
}

void subindex_local(int n, const Particle *pp, /**/ Clist *c, Ticket *t) {
    subindex(n, pp, c->dims, /**/ c->counts, t->eelo);
}

void subindex_remote(int n, const Particle *pp, /**/ Clist *c, Ticket *t) {
    subindex(n, pp, c->dims, /**/ c->counts, t->eere);
}




void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, Ticket *t) {
    build(nlo, 0, nout, pplo, NULL, /**/ ppout, c, t);
}

void build(int nlo, int nre, int nout, const Particle *pplo, const Particle *ppre, /**/ Particle *ppout, Clist *c, Ticket *t) {
    int nc, *cc, *ss;
    uchar4 *eelo, *eere;
    int3  dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;
    eelo = t->eelo;
    eere = t->eere;
    uint *ii = t->ii;
    
    CC(d::MemsetAsync(cc, 0, nc * sizeof(int)));

    if (nlo) KL(dev::subindex, (k_cnf(nlo)), (dims, nlo, pplo, /*io*/ cc, /**/ eelo));
    if (nre) KL(dev::subindex, (k_cnf(nre)), (dims, nre, ppre, /*io*/ cc, /**/ eere));

    scan::scan(cc, nc, /**/ ss, /*w*/ &t->scan);

    if (nlo) KL(dev::get_ids, (k_cnf(nlo)), (LOCAL,  dims, nlo, ss, eelo, /**/ ii));
    if (nre) KL(dev::get_ids, (k_cnf(nre)), (REMOTE, dims, nre, ss, eere, /**/ ii));

    KL(dev::gather, (k_cnf(nout)), (pplo, ppre, ii, nout, /**/ ppout));
}

#undef REMOTE
#undef LOCAL

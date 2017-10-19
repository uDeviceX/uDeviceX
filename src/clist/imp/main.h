enum {LOCAL, REMOTE};

void ini_counts(/**/ Clist *c) {
    if (c->ncells) CC(d::MemsetAsync(c->counts, 0, c->ncells * sizeof(int)));
}

static void subindex(int n, const Particle *pp, int3 dims, /**/ int *cc, uchar4 *ee) {
    if (n) KL(dev::subindex, (k_cnf(n)), (dims, n, pp, /*io*/ cc, /**/ ee));
}

void subindex_local(int n, const Particle *pp, /**/ Clist *c, Map *m) {
    subindex(n, pp, c->dims, /**/ c->counts, m->eelo);
}

void subindex_remote(int n, const Particle *pp, /**/ Clist *c, Map *m) {
    subindex(n, pp, c->dims, /**/ c->counts, m->eere);
}

void build_map(int nlo, int nre, /**/ Clist *c, Map *m) {
    int nc, *cc, *ss;
    uchar4 *eelo, *eere;
    uint *ii = m->ii;
    int3 dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;
    eelo = m->eelo;
    eere = m->eere;
        
    scan::scan(cc, nc, /**/ ss, /*w*/ &m->scan);
   
    if (nlo) KL(dev::get_ids, (k_cnf(nlo)), (LOCAL,  dims, nlo, ss, eelo, /**/ ii));
    if (nre) KL(dev::get_ids, (k_cnf(nre)), (REMOTE, dims, nre, ss, eere, /**/ ii));    
}

void gather_pp(const Particle *pplo, const Particle *ppre, const Map *m, int nout, /**/ Particle *ppout) {
    Sarray <const Particle*, 2> src = {pplo, ppre};
    if (nout) KL(dev::gather, (k_cnf(nout)), (src, m->ii, nout, /**/ ppout));
}

void gather_ii(const int *iilo, const int *iire, const Map *m, int nout, /**/ int *iiout) {
    Sarray <const int*, 2> src = {iilo, iire};
    if (nout) KL(dev::gather, (k_cnf(nout)), (src, m->ii, nout, /**/ iiout));
}

void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, Map *m) {
    ini_counts(/**/ c);
    subindex_local (nlo, pplo, /**/ c, m);
    build_map(nlo, 0, /**/ c, m);    
    gather_pp(pplo, NULL, m, nout, ppout);
}

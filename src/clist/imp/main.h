#define REMOTE (true)
#define LOCAL (false)

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

    /* used for debugging purpose */    
    // if (nlo + nre) KL(dev::ini_ids, (k_cnf(nlo+nre)), (nlo+nre, /**/ ii));
   
    if (nlo) KL(dev::get_ids, (k_cnf(nlo)), (LOCAL,  dims, nlo, ss, eelo, /**/ ii));
    if (nre) KL(dev::get_ids, (k_cnf(nre)), (REMOTE, dims, nre, ss, eere, /**/ ii));    
}

void gather_pp(const Particle *pplo, const Particle *ppre, const Map *m, int nout, /**/ Particle *ppout) {
    if (nout) KL(dev::gather, (k_cnf(nout)), (pplo, ppre, m->ii, nout, /**/ ppout));
}

void gather_ii(const int *iilo, const int *iire, const Map *m, int nout, /**/ int *iiout) {
    if (nout) KL(dev::gather, (k_cnf(nout)), (iilo, iire, m->ii, nout, /**/ iiout));
}

static void build(int nlo, int nre, int nout, const Particle *pplo, const Particle *ppre, /**/ Particle *ppout, Clist *c, Map *m) {
    ini_counts(/**/ c);
    subindex_local (nlo, pplo, /**/ c, m);
    subindex_remote(nre, ppre, /**/ c, m);
    build_map(nlo, nre, /**/ c, m);    
    gather_pp(pplo, ppre, m, nout, ppout);
}

void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, Map *m) {
    build(nlo, 0, nout, pplo, NULL, /**/ ppout, c, m);
}


#undef REMOTE
#undef LOCAL

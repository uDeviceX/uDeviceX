enum {LOCAL, REMOTE};

void ini_counts(/**/ Clist *c) {
    if (c->ncells) CC(d::MemsetAsync(c->counts, 0, c->ncells * sizeof(int)));
}

static void subindex(int n, const Particle *pp, int3 dims, /**/ int *cc, uchar4 *ee) {
    if (n) KL(dev::subindex, (k_cnf(n)), (dims, n, pp, /*io*/ cc, /**/ ee));
}

void subindex_local(int n, const Particle *pp, /**/ Clist *c, Map *m) {
    subindex(n, pp, c->dims, /**/ c->counts, m->ee[LOCAL]);
}

void subindex_remote(int n, const Particle *pp, /**/ Clist *c, Map *m) {
    subindex(n, pp, c->dims, /**/ c->counts, m->ee[REMOTE]);
}

void build_map(const int nn[], /**/ Clist *c, Map *m) {
    int nc, *cc, *ss, n, i;
    const uchar4 *ee;
    uint *ii = m->ii;
    int3 dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;
        
    scan::scan(cc, nc, /**/ ss, /*w*/ &m->scan);

    for (i = 0; i < m->nA; ++i) {
        n = nn[i];
        ee = m->ee[i];
        if (n) KL(dev::get_ids, (k_cnf(n)), (i, dims, n, ss, ee, /**/ ii));
    }
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
    const int nn[] = {nlo, 0};
    ini_counts(/**/ c);
    subindex_local (nlo, pplo, /**/ c, m);
    build_map(nn, /**/ c, m);    
    gather_pp(pplo, NULL, m, nout, ppout);
}

enum {LOCAL, REMOTE};

const uint* clist_get_ids(const ClistMap *m) {return m->ii;}

void clist_ini_counts(/**/ Clist *c) {
    if (c->ncells) CC(d::MemsetAsync(c->counts, 0, (c->ncells + 16) * sizeof(int)));
}

static void check_input_size(int id, long n, const ClistMap *m) {
    long cap = m->maxp;
    if (n > cap)
        ERR("Too many input particles for array %d (%ld / %ld)", id, n, cap);
}

static void comp_subindices(bool project, int n, const PartList lp, int3 dims, /**/ int *cc, uchar4 *ee) {
    if (n) KL(clist_dev::subindex, (k_cnf(n)), (project, dims, n, lp, /*io*/ cc, /**/ ee));
}

void clist_subindex(bool project, int aid, int n, const PartList lp, /**/ Clist *c, ClistMap *m) {
    UC(check_input_size(aid, n, m));
    UC(comp_subindices(project, n, lp, c->dims, /**/ c->counts, m->ee[aid]));
}

void clist_subindex_local(int n, const PartList lp, /**/ Clist *c, ClistMap *m) {
    UC(clist_subindex(false, LOCAL, n, lp, /**/ c, m));
}

void clist_subindex_remote(int n, const PartList lp, /**/ Clist *c, ClistMap *m) {
    UC(clist_subindex(false, REMOTE, n, lp, /**/ c, m));
}

void clist_build_map(const int nn[], /**/ Clist *c, ClistMap *m) {
    int nc, *cc, *ss, n, i;
    const uchar4 *ee;
    uint *ii = m->ii;
    int3 dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;
        
    scan_apply(cc, nc + 16, /**/ ss, /*w*/ m->scan);

    for (i = 0; i < m->nA; ++i) {
        n = nn[i];
        ee = m->ee[i];
        UC(check_input_size(i, n, m));
        if (n) KL(clist_dev::get_ids, (k_cnf(n)), (i, dims, n, ss, ee, /**/ ii));
    }
}

static void check_map_capacity(long n, const ClistMap *m) {
    long cap = m->maxp * m->nA;
    if (n > cap)
        ERR("Too many particles for this cell list (%ld / %ld)", n, cap);     
}

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout, /**/ Particle *ppout) {
    Sarray <const Particle*, 2> src = {pplo, ppre};
    UC(check_map_capacity(nout, m));
    if (nout)
        KL(clist_dev::gather, (k_cnf(nout)), (src, m->ii, nout, /**/ ppout));
}

void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout, /**/ int *iiout) {
    Sarray <const int*, 2> src = {iilo, iire};
    UC(check_map_capacity(nout, m));    
    if (nout)
        KL(clist_dev::gather, (k_cnf(nout)), (src, m->ii, nout, /**/ iiout));
}

void clist_build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, ClistMap *m) {
    const int nn[] = {nlo, 0};
    PartList lp;
    lp.pp = pplo;
    lp.deathlist = NULL;
    UC(clist_ini_counts(/**/ c));
    UC(clist_subindex_local (nlo, lp, /**/ c, m));
    UC(clist_build_map(nn, /**/ c, m));
    UC(clist_gather_pp(pplo, NULL, m, nout, ppout));
}

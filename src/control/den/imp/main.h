static void reset_ndead(DCont *d) {
    CC(d::MemsetAsync(d->ndead_dev, 0, sizeof(int)));
    d->ndead = 0;
}

void ini(int maxp, /**/ DCont **d0) {
    DCont *d;
    size_t sz;
    
    UC(emalloc(sizeof(DCont), (void**) d0));
    d = *d0;

    sz = maxp * sizeof(int);
    CC(d::Malloc((void**) &d->kk, sz));
    CC(d::Malloc((void**) &d->ndead_dev, sizeof(int)));

    CC(d::MemsetAsync(d->kk, 0, sz));
    reset_ndead(d);
}

void fin(DCont *d) {
    CC(d::Free(d->kk));
    CC(d::Free(d->ndead_dev));
    UC(efree(d));
}

void reset(int n, /**/ DCont *d) {
    reset_ndead(d);
    CC(d::MemsetAsync(d->kk, 0, n * sizeof(int)));
}

void filter_particles(const DContMap *m, const int *starts, const int *counts, /**/ DCont *d) {    
    KL( kill, (k_cnf(m->n)), (numberdensity, starts, counts, m->n, m->cids, /**/ d->ndead_dev, d->kk) );
}

void download_ndead(DCont *d) {
    CC(d::Memcpy(&d->ndead, d->ndead_dev, sizeof(int), D2H));
    // msg_print("killed %d particles", o->ndead);
}


int* get_deathlist(DCont *d) {return d->kk;}
int  get_ndead(DCont *d)     {return d->ndead;}

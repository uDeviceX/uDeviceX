void ini_map(int nfrags, const int capacity[], /**/ Map *m) {
    CC(d::Malloc((void**) &m->counts,  nfrags      * sizeof(int)));
    CC(d::Malloc((void**) &m->starts, (nfrags + 1) * sizeof(int)));

    int i, c;
    for (i = 0; i < nfrags; ++i) {
        c = capacity[i];
        if (c) CC(d::Malloc((void**) &m->ids[i], c * sizeof(int)));
    }
}

void fin_map(int nfrags, /**/ Map *m) {
    CC(d::Free(m->counts));
    CC(d::Free(m->starts));    
    for (int i = 0; i < nfrags; ++i)
        CC(d::Free(m->ids[i]));
}

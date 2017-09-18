void ini_map(const int capacity[NBAGS], /**/ Map *m) {
    CC(d::Malloc((void**) &m->counts,  NFRAGS      * sizeof(int)));
    CC(d::Malloc((void**) &m->starts, (NFRAGS + 1) * sizeof(int)));

    for (int i = 0; i < NFRAGS; ++i)
        CC(d::Malloc((void**) &m->ids[i], capacity[i] * sizeof(int)));
}

void fin_map(/**/ Map *m) {
    CC(d::Free(m->counts));
    CC(d::Free(m->starts));    
    for (int i = 0; i < NFRAGS; ++i)
        CC(d::Free(m->ids[i]));
}

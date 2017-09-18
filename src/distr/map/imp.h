void ini_map(const int capacity[NBAGS], /**/ Map *m) {
    CC(d::Malloc((void**) &m->counts,  NBAGS      * sizeof(int)));
    CC(d::Malloc((void**) &m->starts, (NBAGS + 1) * sizeof(int)));

    int i, c;
    for (i = 0; i < NBAGS; ++i) {
        c = capacity[i];
        if (c) CC(d::Malloc((void**) &m->ids[i], c * sizeof(int)));
        else m->ids[i] = NULL;
    }
}

void fin_map(/**/ Map *m) {
    CC(d::Free(m->counts));
    CC(d::Free(m->starts));    
    for (int i = 0; i < NBAGS; ++i)
        if (m->ids[i])
            CC(d::Free(m->ids[i]));
}

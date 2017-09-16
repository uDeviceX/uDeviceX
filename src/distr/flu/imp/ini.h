
void alloc_map(float maxdensity, /**/ Map *m) {
    CC(d::Malloc((void**) &m->counts,  NFRAGS      * sizeof(int)));
    CC(d::Malloc((void**) &m->starts, (NFRAGS + 1) * sizeof(int)));

    int e[NFRAGS], i;
    frag_estimates(NFRAGS, maxdensity, /**/ e);

    for (i = 0; i < NFRAGS; ++i)
        CC(d::Malloc((void**) &m->ids[i], e[i] * sizeof(int)));
}


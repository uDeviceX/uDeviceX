// TODO: belongs to fragment?
static void estimates(int nfrags, float maxd, /**/ int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = frag_ncell(i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}

void alloc_map(float maxdensity, /**/ Map *m) {
    enum {NFRAGS = 26};
    CC(d::Malloc((void**) &m->counts,  NFRAGS      * sizeof(int)));
    CC(d::Malloc((void**) &m->starts, (NFRAGS + 1) * sizeof(int)));

    int e[NFRAGS], i;
    estimates(NFRAGS, maxdensity, /**/ e);

    for (i = 0; i < NFRAGS; ++i)
        CC(d::Malloc((void**) &m->ids[i], e[i] * sizeof(int)));
}


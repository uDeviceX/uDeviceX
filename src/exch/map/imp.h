void ini_map(int nw, int nfrags, int cap[], /**/ Map *map) {
    int i, c;
    size_t sz;
    sz = (nw + 1) * (nfrags + 1) * sizeof(int);
    CC(d::Malloc((void**) &map->counts,  sz));
    CC(d::Malloc((void**) &map->starts,  sz));
    CC(d::Malloc((void**) &map->offsets, sz));

    for (i = 0; i < nfrags; ++i) {
        c = cap[i];
        sz = c * sizeof(int);
        CC(d::Malloc((void**) &map->ids[i], sz));
    }
}

void fin_map(int nfrags, Map *map) {
    CC(d::Free(map->counts));
    CC(d::Free(map->starts));
    CC(d::Free(map->offsets));

    for (int i = 0; i < nfrags; ++i)
        CC(d::Free(map->ids[i]));
}


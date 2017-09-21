static void reini_map(int nw, int nfrags, /**/ Map map) {
    size_t sz;
    sz = (nw + 1) * (nfrags + 1) * sizeof(int);
    CC(d::cudaMemsetAsync(map->counts,  0, sz));
    CC(d::cudaMemsetAsync(map->starts,  0, sz));
    CC(d::cudaMemsetAsync(map->offsets, 0, sz));
}

static void add_wrap_to_map(int wid, int n, const Particle *pp, Map map) {
    int3 L(XS-2, YS-2, ZS-2);    
    KL(dev::build_map, (k_cnf(n)), (L, wid, pp, n, /**/ map));
}

static void fill_map(int nw, const ParticleWrap *ww, /**/ Map *map) {
    ParticleWrap *w;
    for (int i = 0; i < nw; ++i) {
        w = ww + i;
        add_wrap_to_map(i, w->n, w->pp, /**/ map);
    }
}

static void scan_map(int nw, int nfrags, /**/ Map map) {
    // TODO
}

void build_map(int nw, const ParticleWrap *ww, /**/ Map *map) {
    reini_map(nw, NFRAGS, /**/ map);
    fill_map(nw, ww, /**/ map);
    scan_map(nw, NFRAGS, /**/ map);
}

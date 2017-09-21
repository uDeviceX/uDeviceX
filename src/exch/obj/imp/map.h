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

static void fill_map(int nw, const ParticlesWrap *ww, /**/ Map *map) {
    ParticlesWrap *w;
    for (int i = 0; i < nw; ++i) {
        w = ww + i;
        add_wrap_to_map(i, w->n, w->pp, /**/ map);
    }
}

static void scan_map(int nw, int nfrags, /**/ Map map) {
    int i, *cc, *ss, *oo, *oon, stride;
    stride = nfrags + 1;
    for (i = 0; i < nw; ++i) {
        cc  = map.counts  + i * stride;
        ss  = map.starts  + i * stride;
        oo  = map.offsets + i * stride;
        oon = map.offsets + (i + 1) * stride;
        KL(dev::scan2d, (1, 32), (cc, oo, /**/ oon, ss));
    }
    KL(dev::scan2d, (1, 32), (oon, /**/ map.totstarts));
}

void build_map(int nw, const ParticlesWrap *ww, /**/ Map *map) {
    reini_map(nw, NFRAGS, /**/ map);
    fill_map(nw, ww, /**/ map);
    scan_map(nw, NFRAGS, /**/ map);
}

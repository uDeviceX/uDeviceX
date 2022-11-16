static void add_wrap_to_map(int3 L, int wid, int n, const Particle *pp, EMap map) {
    int3 L0 = make_int3(L.x-2, L.y-2, L.z-2);    
    KL(eobj_dev::build_map, (k_cnf(n)), (L0, wid, n, pp, /**/ map));
}

static void fill_map(int3 L, int nw, const PaWrap *ww, /**/ EMap map) {
    const PaWrap *w;
    for (int i = 0; i < nw; ++i) {
        w = ww + i;
        add_wrap_to_map(L, i, w->n, w->pp, /**/ map);
    }
}

static void build_map(int3 L, int nw, const PaWrap *ww, /**/ EMap map) {
    emap_reini(nw, NFRAGS, /**/ map);
    fill_map(L, nw, ww, /**/ map);
    emap_scan(nw, NFRAGS, /**/ map);
}

void eobj_build_map(int nw, const PaWrap *ww, /**/ EObjPack *p) {
    if (!nw) return;
    build_map(p->L, nw, ww, /**/ p->map);
}

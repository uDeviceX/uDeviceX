static void add_wrap_to_map(int wid, int n, const Particle *pp, EMap map) {
    int3 L = make_int3(XS-2, YS-2, ZS-2);    
    KL(dev::build_map, (k_cnf(n)), (L, wid, n, pp, /**/ map));
}

static void fill_map(int nw, const PaWrap *ww, /**/ EMap map) {
    const PaWrap *w;
    for (int i = 0; i < nw; ++i) {
        w = ww + i;
        add_wrap_to_map(i, w->n, w->pp, /**/ map);
    }
}

static void build_map(int nw, const PaWrap *ww, /**/ EMap map) {
    emap_reini(nw, NFRAGS, /**/ map);
    fill_map(nw, ww, /**/ map);
    emap_scan(nw, NFRAGS, /**/ map);
}

void eobj_build_map(int nw, const PaWrap *ww, /**/ EObjPack *p) {
    if (!nw) return;
    build_map(nw, ww, /**/ p->map);
}

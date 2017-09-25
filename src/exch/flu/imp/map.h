
static void fill_map(int n, const Particle *pp, /**/ Map map) {
    int3 L = make_int3(XS-2, YS-2, ZS-2);    
    KL(dev::build_map, (k_cnf(n)), (L, 0, n, pp, /**/ map));
}

static void build_map(int n, const Particle *pp, /**/ Map map) {
    reini_map(1, NFRAGS, /**/ map);
    fill_map(n, pp, /**/ map);
    scan_map(1, NFRAGS, /**/ map);
}

void build_map(int n, const Particle *pp, /**/ Pack *p) {
    build_map(n, pp, /**/ p->map);
}


static void fill_map(int n, const Particle *pp, const float3 *lo, const float3 *hi, /**/ Map map) {
    int3 L = make_int3(XS-2, YS-2, ZS-2);
    KL(dev::build_map, (k_cnf(n)), (L, n, lo, hi, /**/ map));
}

void build_map(int nm, int nv, const Particle *pp, /**/ Map map) {
    reini_map(1, NFRAGS, /**/ p->map);
    minmax(pp, nv, nm, /**/ p->minext, p->maxext);
    fill_map(nm * nv, p->minext, p->maxext, /**/ p->map);
    scan_map(1, NFRAGS, /**/ p->map);
}

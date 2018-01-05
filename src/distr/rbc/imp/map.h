static void build_map(int n, const float3 *minext, const float3 *maxext, /**/ Map m) {
    map_reini(NBAGS, /**/ m);
    KL(dev::build_map, (k_cnf(n)), (n, minext, maxext, /**/ m));
    KL(dev::scan_map<NBAGS>, (1, 32), (/**/ m));
}

void build_map(int nc, int nv, const Particle *pp, Pack *p) {
    minmax(pp, nv, nc, /**/ p->minext, p->maxext);
    build_map(nc, p->minext, p->maxext, /**/  p->map);
}

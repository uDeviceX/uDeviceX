static void build_map(int n, const float3 *minext, const float3 *maxext, /**/ DMap m) {
    UC(dmap_reini(NBAGS, /**/ m));
    KL(dev::build_map, (k_cnf(n)), (n, minext, maxext, /**/ m));
    KL(dmap_scan<NBAGS>, (1, 32), (/**/ m));
}

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p) {
    minmax(pp, nv, nc, /**/ p->minext, p->maxext);
    build_map(nc, p->minext, p->maxext, /**/  p->map);
}

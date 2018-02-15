static void build_map(int3 L, int n, const float3 *minext, const float3 *maxext, /**/ DMap m) {
    UC(dmap_reini(NBAGS, /**/ m));
    KL(drbc_dev::build_map, (k_cnf(n)), (L, n, minext, maxext, /**/ m));
    KL(dmap_scan<NBAGS>, (1, 32), (/**/ m));
}

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p) {
    minmax(pp, nv, nc, /**/ p->minext, p->maxext);
    build_map(p->L, nc, p->minext, p->maxext, /**/  p->map);
}

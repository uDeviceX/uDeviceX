static void fill_map(int3 L, int n, const float3 *lo, const float3 *hi, /**/ EMap map) {
    int3 L0 = make_int3(L.x-2, L.y-2, L.z-2);
    KL(emesh_dev::build_map, (k_cnf(n)), (L0, n, lo, hi, /**/ map));
}

void emesh_build_map(int nm, int nv, const Particle *pp, /**/ EMeshPack *p) {
    emap_reini(1, NFRAGS, /**/ p->map);
    minmax(pp, nv, nm, /**/ p->minext, p->maxext);
    fill_map(p->L, nm, p->minext, p->maxext, /**/ p->map);
    emap_scan(1, NFRAGS, /**/ p->map);
}

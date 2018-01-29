static void build_map(int3 L, int n, const Solid *ss, /**/ DMap m) {
    UC(dmap_reini(NBAGS, /**/ m));
    KL(dev::build_map, (k_cnf(n)), (L, n, ss, /**/ m));
    KL(dmap_scan<NBAGS>, (1, 32), (/**/ m));
}

void drig_build_map(int ns, const Solid *ss, /**/ DRigPack *p) {
    UC(build_map(p->L, ns, ss, /**/ p->map));
}

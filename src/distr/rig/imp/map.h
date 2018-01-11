static void build_map(int n, const Solid *ss, /**/ DMap m) {
    UC(dmap_reini(NBAGS, /**/ m));
    KL(dev::build_map, (k_cnf(n)), (n, ss, /**/ m));
    KL(dmap_scan<NBAGS>, (1, 32), (/**/ m));
}

void drig_build_map(int ns, const Solid *ss, /**/ Pack *p) {
    UC(build_map(ns, ss, /**/ p->map));
}

static void build_map(int n, const Particle *pp, Map m) {
    reini_map(NFRAGS, /**/ m);
    KL(dev::build_map, (k_cnf(n)), (pp, n, /**/ m));
    KL(dev::scan_map<NFRAGS>, (1, 32), (/**/ m));
}

void build_map(int n, const Particle *pp, Pack *p) {
    build_map(n, pp, p->map);
}

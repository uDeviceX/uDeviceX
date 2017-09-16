static void reini_map(Map m) {
    CC(d::MemsetAsync(m.counts, 0, NFRAGS * sizeof(int)));
}

static void build_map(int n, const Particle *pp, Map m) {
    reini_map(/**/ m);
    KL(dev::build_map, (k_cnf(n)), (pp, n, /**/ m));
    KL(dev::scan_map<NFRAGS>, (1, 32), (/**/ m));
}

void build_map(int n, const Particle *pp, Pack *p) {
    build_map(n, pp, p->map);
}

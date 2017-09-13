static void reini_map(Map *m) {
    CC(d::MemsetAsync(m->counts, 0, 26 * sizeof(int)));
}

void build_map(int n, const Particle *pp, Map *m) {
    reini_map(/**/ m);
    KL(dev::build_map, (k_cnf(n)), (pp, n, /**/ m));
    KL(dev::scan_map, (1, 32), (/**/ m));    
}

void pack(const Map *m, Particle *pp, /**/ dBags *bags) {
    
}

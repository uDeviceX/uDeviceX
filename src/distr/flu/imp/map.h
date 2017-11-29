static void build_map(int n, const PartList lp, Map m) {
    reini_map(NFRAGS, /**/ m);
    KL(dev::build_map, (k_cnf(n)), (lp, n, /**/ m));
    download_counts(NFRAGS, /**/ &m); /* async */
    KL(dev::scan_map<NFRAGS>, (1, 32), (/**/ m));
}

void build_map(int n, const PartList lp, Pack *p) {
    build_map(n, lp, p->map);
    dSync();
}

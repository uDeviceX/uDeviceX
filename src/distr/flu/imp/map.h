static void build_map(int n, const PartList lp, Map m) {
    UC(map_reini(NFRAGS, /**/ m));
    KL(dev::build_map, (k_cnf(n)), (lp, n, /**/ m));
    UC(map_download_counts(NFRAGS, /**/ &m)); /* async */
    KL(dev::scan_map<NFRAGS>, (1, 32), (/**/ m));
}

void build_map(int n, const PartList lp, Pack *p) {
    UC(build_map(n, lp, p->map));
    dSync();
}

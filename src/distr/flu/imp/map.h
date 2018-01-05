static void build_map(int n, const PartList lp, DMap m) {
    UC(dmap_reini(NFRAGS, /**/ m));
    KL(dev::build_map, (k_cnf(n)), (lp, n, /**/ m));
    UC(dmap_download_counts(NFRAGS, /**/ &m)); /* async */
    KL(dmap_scan<NFRAGS>, (1, 32), (/**/ m));
}

void build_map(int n, const PartList lp, Pack *p) {
    UC(build_map(n, lp, p->map));
    dSync();
}

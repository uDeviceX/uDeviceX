static void build_map(int3 L, int n, const PartList lp, DMap m) {
    UC(dmap_reini(NFRAGS, /**/ m));
    KL(dflu_dev::build_map, (k_cnf(n)), (L, lp, n, /**/ m));
    UC(dmap_download_counts(NFRAGS, /**/ &m)); /* async */
    KL(dmap_scan<NFRAGS>, (1, 32), (/**/ m));
}

void dflu_build_map(int n, const PartList lp, DFluPack *p) {
    UC(build_map(p->L, n, lp, p->map));
    dSync();
}

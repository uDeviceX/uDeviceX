/* map */

void build_map(int n, const Particle *pp, Pack *p) {
    build_map(n, pp, p->map);
    dSync();
}

/* pack */

void pack_pp(const Particle *pp, int n, /**/ Pack *p) {
    pack_pp(p->map, pp, n, /**/ p->dpp);
}

void pack_ii(const int *ii, int n, /**/ Pack *p) {
    pack_ii(p->map, ii, n, /**/ p->dii);
}

void pack_cc(const int *cc, int n, /**/ Pack *p) {
    pack_ii(p->map, cc, n, /**/ p->dcc);
}

void download(int n, Pack *p) {
    CC(d::Memcpy(p->hpp.counts, p->map.counts, 26 * sizeof(int), D2H));
    if (global_ids)    CC(d::Memcpy(p->hii.counts, p->map.counts, 26 * sizeof(int), D2H));
    if (multi_solvent) CC(d::Memcpy(p->hcc.counts, p->map.counts, 26 * sizeof(int), D2H));

    int nhalo, i, c;
    for (i = nhalo = 0; i < NFRAGS; ++i) {
        c = p->hpp.counts[i];
        nhalo += c;
    }
    p->nbulk = n - nhalo;
}

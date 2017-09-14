/* map */

void build_map(int n, const Particle *pp, Pack *p) {
    build_map(n, pp, p->map);
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

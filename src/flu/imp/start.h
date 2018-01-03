#define CODE "flu"
#define COLEXT "colors"
#define IDSEXT "ids"

static int strt_pp(Coords coords, const int id, Particle *dev, /*w*/ Particle *hst) {
    int n;
    restart::read_pp(coords, CODE, id, hst, &n);
    if (n) cH2D(dev, hst, n);
    return n;
}

static int strt_ii(Coords coords, const char *subext, const int id, int *dev, /*w*/ int *hst) {
    int n;
    restart::read_ii(coords, CODE, subext, id, hst, &n);
    if (n) cH2D(dev, hst, n);
    return n;
}

void strt_quants(Coords coords, const int id, Quants *q) {
    q->n =             strt_pp(coords,         id, /**/ q->pp, /* w */ q->pp_hst);
    if (global_ids)    strt_ii(coords, IDSEXT, id, /**/ q->ii, /* w */ q->ii_hst);
    if (multi_solvent) strt_ii(coords, COLEXT, id, /**/ q->cc, /* w */ q->cc_hst);
}

static void strt_dump_pp(Coords coords, const int id, const int n, const Particle *dev, Particle *hst) {
    if (n) cD2H(hst, dev, n);
    restart::write_pp(coords, CODE, id, hst, n);
}

static void strt_dump_ii(Coords coords, const char *subext, const int id, const int n, const int *dev, int *hst) {
    if (n) cD2H(hst, dev, n);
    restart::write_ii(coords, CODE, subext, id, hst, n);
}

void strt_dump(Coords coords, const int id, const Quants q) {
    strt_dump_pp(coords, id, q.n, q.pp, /* w */ q.pp_hst);
    if (global_ids)    strt_dump_ii(coords, IDSEXT, id, q.n, q.ii, /* w */ q.ii_hst);
    if (multi_solvent) strt_dump_ii(coords, COLEXT, id, q.n, q.cc, /* w */ q.cc_hst);
}

#undef CODE
#undef COLEXT
#undef IDSEXT

#define CODE "flu"
#define COLEXT "colors"
#define IDSEXT "ids"

static int strt_pp(const Coords *coords, const int id, Particle *dev, /*w*/ Particle *hst) {
    int n;
    restart::restart_read_pp(coords, CODE, id, hst, &n);
    if (n) cH2D(dev, hst, n);
    return n;
}

static int strt_ii(const Coords *coords, const char *subext, const int id, int *dev, /*w*/ int *hst) {
    int n;
    restart::restart_read_ii(coords, CODE, subext, id, hst, &n);
    if (n) cH2D(dev, hst, n);
    return n;
}

void flu_strt_quants(const Coords *coords, const int id, FluQuants *q) {
    q->n =             strt_pp(coords,         id, /**/ q->pp, /* w */ q->pp_hst);
    if (global_ids)    strt_ii(coords, IDSEXT, id, /**/ q->ii, /* w */ q->ii_hst);
    if (multi_solvent) strt_ii(coords, COLEXT, id, /**/ q->cc, /* w */ q->cc_hst);
}

static void strt_dump_pp(const Coords *coords, const int id, const int n, const Particle *dev, Particle *hst) {
    if (n) cD2H(hst, dev, n);
    restart::restart_write_pp(coords, CODE, id, hst, n);
}

static void strt_dump_ii(const Coords *coords, const char *subext, const int id, const int n, const int *dev, int *hst) {
    if (n) cD2H(hst, dev, n);
    restart::restart_write_ii(coords, CODE, subext, id, hst, n);
}

void flu_strt_dump(const Coords *coords, const int id, const FluQuants *q) {
    strt_dump_pp(coords, id, q->n, q->pp, /* w */ q->pp_hst);
    if (global_ids)    strt_dump_ii(coords, IDSEXT, id, q->n, q->ii, /* w */ q->ii_hst);
    if (multi_solvent) strt_dump_ii(coords, COLEXT, id, q->n, q->cc, /* w */ q->cc_hst);
}

#undef CODE
#undef COLEXT
#undef IDSEXT

static int strt(const int id, Particle *dev, /*w*/ Particle *hst) {
    int n;
    restart::read_pp("flu", id, hst, &n);
    if (n) cH2D(dev, hst, n);
    return n;
}

void strt_quants(const int id, Quants *q) {
    q->n = strt(id, /**/ q->pp, /* w */ q->pp_hst);
}

static int strt_ii(const char *subext, const int id, int *dev, /*w*/ int *hst) {
    int n;
    restart::read_ii("flu", subext, id, hst, &n);
    if (n) cH2D(dev, hst, n);
    return n;
}

void strt_ii(const char *subext, const int id, QuantsI *q) {
    strt_ii(subext, id, /**/ q->ii, /* w */ q->ii_hst);
}

static void strt_dump(const int id, const int n, const Particle *dev, Particle *hst) {
    if (n) cD2H(hst, dev, n);
    restart::write_pp("flu", id, hst, n);
}

void strt_dump(const int id, const Quants q) {
    strt_dump(id, q.n, q.pp, /* w */ q.pp_hst);
}

static void strt_dump_ii(const char *subext, const int id, const int n, const int *dev, int *hst) {
    if (n) cD2H(hst, dev, n);
    restart::write_ii("flu", subext, id, hst, n);
}

void strt_dump_ii(const char *subext, const int id, const QuantsI q, const int n) {
    strt_dump_ii(subext, id, n, q.ii, /* w */ q.ii_hst);
}

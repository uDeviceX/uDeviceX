static void local2global_p(const Coords *c, long n, Particle *pp) {
    enum {X, Y, Z};
    float *r;
    int i;
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        r[X] = xl2xg(c, r[X]);
        r[Y] = yl2yg(c, r[Y]);
        r[Z] = zl2zg(c, r[Z]);
    }
}

static void gen_name(const Coords *c, /**/ char *name) {
    int r;
    char stamp[FILENAME_MAX];
    coord_stamp(c, /**/ stamp);
    r = sprintf(name, "%s.punto", stamp);
    if (r < 0) ERR("sprintf failed");
}

void flu_punto_dump(const Coords *c, const FluQuants *q) {
    Particle *dev, *hst;
    int n;
    char name[FILENAME_MAX];
    n = q->n; dev = q->pp; hst = q->pp_hst;
    if (n) cD2H(hst, dev, n);
    local2global_p(c, n, /**/ hst);
    UC(gen_name(c, /**/ name));
    UC(punto_dump_pp(n, hst, name));
}

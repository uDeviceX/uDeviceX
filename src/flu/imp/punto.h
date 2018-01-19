static void local2global_p(Coords *c, long n, Particle *pp) {
    enum {X, Y, Z};
    float *r;
    int i;
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        r[X] = xl2xg(*c, r[X]);
        r[Y] = yl2yg(*c, r[Y]);
        r[Z] = zl2zg(*c, r[Z]);
    }
}

void flu_punto_dump(Coords *c, const FluQuants *q) {
    Particle *dev, *hst;
    int n;
    n = q->n; dev = q->pp; hst = q->pp_hst;
    if (n) cD2H(hst, dev, n);
    local2global_p(c, n, /**/ hst);
    UC(punto_dump(n, hst, "preved.punto"));
}

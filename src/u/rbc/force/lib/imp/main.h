static void write0(Particle p, Force f0) {
    enum {X, Y, Z};
    float *r, *f;
    r = p.r;
    f = f0.f;
    printf("%g %g %g %g %g %g\n", r[X], r[Y], r[Z], f[X], f[Y], f[Z]);
}

static void write1(int n, Particle *p, Force *f) {
    int i;
    for (i = 0; i < n; i++) write0(p[i], f[i]);
}

static void write(int n, Particle *p, Force *f) {
    Particle *p_hst;
    Force *f_hst;

    UC(emalloc(n*sizeof(Particle), (void**) &p_hst));
    UC(emalloc(n*sizeof(Force),    (void**) &f_hst));

    cD2H(p_hst, p, n);
    cD2H(f_hst, f, n);

    write1(n, p_hst, f_hst);

    free(p_hst);
    free(f_hst);
}

static void run0(RbcQuants q, RbcForce t, const RbcParams *par, Force *f) {
    rbc_force_apply(q, t, par, /**/ f);
    write(q.n, q.pp, f);
}

static void run1(RbcQuants q, RbcForce t, const RbcParams *par) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);

    run0(q, t, par, f);
    Dfree(f);
}

static void run2(const char *cell, const char *ic, const RbcParams *par, RbcQuants q) {
    Coords *coords;
    RbcForce t;
    OffRead *off;
    UC(off_read(cell, /**/ &off));
    coords_ini(m::cart, XS, YS, ZS, &coords);
    rbc_gen_quants(coords, m::cart, cell, off, ic, /**/ &q);
    rbc_force_gen(q, &t);
    run1(q, t, par);
    rbc_force_fin(&t);
    coords_fin(coords);
    UC(off_fin(off));
}

void run(const char *cell, const char *ic, const RbcParams *par) {
    RbcQuants q;
    rbc_ini(&q);
    run2(cell, ic, par, q);
    rbc_fin(&q);
}

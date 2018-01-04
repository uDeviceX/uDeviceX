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

static void run0(rbc::Quants q, rbc::force::TicketT t, Force *f) {
    rbc::force::apply(q, t, /**/ f);
    write(q.n, q.pp, f);
}

static void run1(rbc::Quants q, rbc::force::TicketT t) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);

    run0(q, t, f);
    Dfree(f);
}

static void run2(const char *cell, const char *ic, rbc::Quants q) {
    Coords coords;
    rbc::force::TicketT t;
    coords_ini(m::cart, &coords);
    rbc::main::gen_quants(coords, m::cart, cell, ic, /**/ &q);
    rbc::force::gen_ticket(q, &t);
    run1(q, t);
    rbc::force::fin_ticket(&t);
    fin_coords(&coords);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run2(cell, ic, q);
    rbc::main::fin(&q);
}

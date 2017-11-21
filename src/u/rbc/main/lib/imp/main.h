static void dump0(rbc::Quants q, const char *f) {
    int n;
    Particle *pp;
    n = q.nc * q.nv;
    UC(emalloc(n*sizeof(Particle), (void**)&pp));
    cD2H(pp, q.pp, q.n);
    io::mesh::main(pp, q.tri_hst, q.nc, q.nv, q.nt, f);
    free(pp);
}

static void dump(rbc::Quants q) {
    static int id = 0;
    char f[BUFSIZ];
    sprintf(f, "%05d.ply", id++);
    MSG("%s",f);
    dump0(q, f);
}

static void run0(rbc::Quants q, rbc::force::TicketT t, Force *f) {
    int i;
    rbc::force::apply(q, t, /**/ f);
    for (i = 0; i < 10; i++) {
        scheme::move(rbc_mass, q.n, f, q.pp);
        if (i % 1 == 0) dump(q);
    }
}

static void run1(rbc::Quants q, rbc::force::TicketT t) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);

    run0(q, t, f);
    Dfree(f);
}

static void run2(const char *cell, const char *ic, rbc::Quants q) {
    rbc::stretch::Fo *stretch;
    rbc::force::TicketT t;
    rbc::main::gen_quants(cell, ic, /**/ &q);
    rbc::stretch::ini("rbc.stretch", q.nv, /**/ &stretch);
    rbc::force::gen_ticket(q, &t);
    run1(q, t);

    rbc::stretch::fin(&stretch);
    rbc::force::fin_ticket(&t);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run2(cell, ic, q);
    rbc::main::fin(&q);
}

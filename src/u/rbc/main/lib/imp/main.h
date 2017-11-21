static void dump(rbc::Quants q) {
    int n;
    Particle *pp;
    static int i = 0;
    n = q.nc * q.nv;
    UC(emalloc(n*sizeof(Particle), (void**)&pp));
    cD2H(pp, q.pp, q.n);
    MSG("dump: %05d", i);
    io::mesh::rbc(pp, q.tri_hst, q.nc, q.nv, q.nt, i++);
    diagnostics(pp, n, i);
    free(pp);
}

static void run0(rbc::Quants q, rbc::force::TicketT t, rbc::stretch::Fo* stretch, Force *f) {
    long i;
    long nsteps = (long)(tend / dt);
    MSG("will take %ld steps", nsteps);
    for (i = 0; i < nsteps; i++) {
        DzeroA(f, q.n);
        rbc::force::apply(q, t, /**/ f);
        rbc::stretch::apply(q.nc, stretch, /**/ f);
        scheme::move(rbc_mass, q.n, f, q.pp);
        if (i % part_freq  == 0) dump(q);
    }
}

static void run1(rbc::Quants q, rbc::force::TicketT t, rbc::stretch::Fo *stretch) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);
    run0(q, t, stretch, f);
    Dfree(f);
}

static void run2(const char *cell, const char *ic, rbc::Quants q) {
    rbc::stretch::Fo *stretch;
    rbc::force::TicketT t;
    rbc::main::gen_quants(cell, ic, /**/ &q);
    rbc::stretch::ini("rbc.stretch", q.nv, /**/ &stretch);
    rbc::force::gen_ticket(q, &t);
    run1(q, t, stretch);
    rbc::stretch::fin(stretch);
    rbc::force::fin_ticket(&t);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run2(cell, ic, q);
    rbc::main::fin(&q);
}

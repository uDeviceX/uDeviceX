static void dump(rbc::Quants q) {
    Particle pp[999999];
    const char *f = "r.ply";
    static int id = 0;
    cD2H(pp, q.pp, q.n);
    io::mesh::main(q.pp, q.tri_hst, q.nc, q.nv, q.nt, f);
}
                 
static void run0(rbc::Quants q, rbc::force::TicketT t, Force *f) {
    rbc::force::apply(q, t, /**/ f);
    //    dump(q);
}

static void run1(rbc::Quants q, rbc::force::TicketT t) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);

    run0(q, t, f);
    Dfree(f);
}

static void run2(const char *cell, const char *ic, rbc::Quants q) {
    rbc::force::TicketT t;
    rbc::main::gen_quants(cell, ic, /**/ &q);
    rbc::force::gen_ticket(q, &t);
    run1(q, t);
    rbc::force::fin_ticket(&t);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run2(cell, ic, q);
    rbc::main::fin(&q);
}

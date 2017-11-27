static void run0(rbc::Quants q, rbc::force::TicketT t) {
    float area, volume, av[2];
    area_volume::hst(q.nt, q.nv, q.nc, q.pp, q.tri, /**/ av);
    area = av[0]; volume = av[1];
    printf("%g %g\n", area, volume);
}

static void run1(const char *cell, const char *ic, rbc::Quants q) {
    rbc::force::TicketT t;
    rbc::main::gen_quants(m::cart, cell, ic, /**/ &q);
    rbc::force::gen_ticket(q, &t);
    run0(q, t);
    rbc::force::fin_ticket(&t);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run1(cell, ic, q);
    rbc::main::fin(&q);
}

static void run0(rbc::Quants q, rbc::force::TicketT t) {
    Force *f;
    MSG("q.n = %d\n", q.n);
    rbc::force::forces(q, t, /**/ f);
}

static void run1(const char *cell, const char *ic, rbc::Quants q) {
    rbc::force::TicketT t;
    rbc::main::gen_quants(cell, ic, /**/ &q);
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

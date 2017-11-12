static void run0(const char *cell, const char *ic, rbc::Quants q) {
    rbc::force::TicketT tt;
    rbc::main::gen_quants(cell, ic, /**/ &q);
    rbc::force::gen_ticket(q, &tt);
    rbc::force::fin_ticket(&tt);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run0(cell, ic, q);
    rbc::main::fin(&q);
}

void fin_rnd(rbc::rnd::D *rnd) {
    rbc::rnd::fin(rnd);
}

void fin_ticket(TicketT *t) {
    destroy(&t->texvert);
    if (RBC_RND) fin_rnd(t->rnd);
}

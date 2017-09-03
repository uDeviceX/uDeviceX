namespace x {
static void fin_tickets() {
    Pfree0(buf_pi);
    fin_ticketcom(tc);
    fin_ticketpack(tp);
    fin_ticketpinned(ti);
}

static void fin_remote() {
    int i;
    rex::RFrag* h;
    for (i = 0; i < 26; i++) {
        h = &remote[i];
        Pfree(h->pp_pi);
        Pfree(h->ff_pi);
    }
}

void fin() {
    rex::fin();
    fin_remote();
    fin_tickets();
}
}

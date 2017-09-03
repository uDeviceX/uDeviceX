namespace x {
static void fin_tickets() {
    Pfree0(buf_pi);
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

static void fin_local() {
    int i;
    LFrag *h;
    for (i = 0; i < 26; i++) {
        h = &local[i];
        Dfree(h->indexes);
        Pfree(h->ff_pi);
    }
}

void fin() {
    rex::fin();
    fin_local();
    fin_remote();
    fin_tickets();
}
}

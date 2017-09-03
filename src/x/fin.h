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
        Pfree(PP_pi.d[i]);
        Pfree(h->ff_pi);
    }
}

static void fin_local() {
    int i;
    rex::LFrag *h;
    for (i = 0; i < 26; i++) {
        h = &local[i];
        Dfree(h->indexes);
        Pfree(h->ff_pi);
    }
}

void fin() {
    fin_local();
    fin_remote();
    fin_tickets();
}
}

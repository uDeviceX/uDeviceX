namespace rex {
static void fin_tickets() {
    Pfree(buf_pi);
    fin_ticketpack(tp);
    fin_ticketpinned(ti);
}

static void fin_remote() {
    int i;
    for (i = 0; i < 26; i++) {
        Pfree(PP_pi.d[i]);
        Pfree(FF_pi.d[i]);
    }
}

static void fin_local() {
    int i;
    sub::LFrag *h;
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

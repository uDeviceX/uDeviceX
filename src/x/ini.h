namespace x {
static int i2max(int i) { /* fragment id to maximum size */
    return MAX_OBJ_DENSITY*frag_ncell(i);
}

static void ini_tickets(/*io*/ basetags::TagGen *g) {
    first = 1;
    ini_ticketcom(&tc);
    ini_ticketr(&tr);
    ini_tickettags(g, &tt);
    ini_ticketpack(&tp);
    ini_ticketpinned(&ti);

    Palloc0(&buf_pi, MAX_PART_NUM);
    Link(&buf, buf_pi);
}

static void ini_remote() {
    int i, n;
    rex::RFrag* h;
    for (i = 0; i < 26; i++) {
        n = i2max(i);
        h = &remote[i];
        Palloc0(&h->pp_pi, n);
        Link(&h->pp, h->pp_pi);

        Palloc0(&h->ff_pi, n);
        Link(&h->ff, h->ff_pi);
    }
}

static void ini_local() {
    int i, n;
    rex::LFrag *h;
    for (i = 0; i < 26; i++) {
        n = i2max(i);
        h = &local[i];
        Dalloc(&h->indexes, n);

        Palloc0(&h->ff_pi, n);
        Link(&h->ff, h->ff_pi);
    }
}

void ini(/*io*/ basetags::TagGen *g) {
    ini_tickets(g);
    ini_local();
    ini_remote();
    rex::ini(local);
}
 
} /* namespace */

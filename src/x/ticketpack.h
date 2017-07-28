namespace x {
static void ini_ticketpack(TicketPack *t) {
    t->tstarts_hst = new PinnedHostBuffer5<int>(27);
    t->offsets_hst = new PinnedHostBuffer5<int>(26);

    Dalloc(&t->tstarts, 27);
    Dalloc(&t->offsets, 27 * (MAX_OBJ_TYPES + 1));
    Dalloc(&t->starts,  27 *  MAX_OBJ_TYPES);
    Dalloc(&t->counts,  26 *  MAX_OBJ_TYPES);
}

static void fin_ticketpack(TicketPack t) {
    delete t.tstarts;
    delete t.tstarts_hst;
    delete t.offsets_hst;

    Dfree(t.tstarts);
    Dfree(t.counts);
    Dfree(t.starts);
    Dfree(t.offsets);
}
}

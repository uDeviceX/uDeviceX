namespace x {
static void ini_ticketpack(TicketPack *t) {
    Dalloc0(&t->tstarts, 27);
    Dalloc0(&t->offsets, 27 * (MAX_OBJ_TYPES + 1));
    Dalloc0(&t->starts,  27 *  MAX_OBJ_TYPES);
    Dalloc0(&t->counts,  26 *  MAX_OBJ_TYPES);
}

static void fin_ticketpack(TicketPack t) {
    Dfree(t.tstarts);
    Dfree(t.counts);
    Dfree(t.starts);
    Dfree(t.offsets);
}
}

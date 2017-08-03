namespace x {
static void ini_ticketpinned(TicketPinned *t) {
    Palloc(&t->tstarts_hst, 27);
    Palloc(&t->offsets_hst, 26);
}

static void fin_ticketpinned(TicketPinned t) {
    Pfree(t.tstarts_hst);
    Pfree(t.offsets_hst);
}
}

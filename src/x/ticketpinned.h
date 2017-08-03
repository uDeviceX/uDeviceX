namespace x {
static void ini_ticketpinned(TicketPinned *t) {
    Palloc(&t->tstarts, 27);
    Palloc(&t->offsets, 26);
}

static void fin_ticketpinned(TicketPinned t) {
    Pfree(t.tstarts);
    Pfree(t.offsets);
}
}

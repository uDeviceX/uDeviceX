namespace x {
static void ini_ticketpinned(TicketPinned *t) {
    Palloc0(&t->tstarts, 27);
    Palloc0(&t->offsets, 26);
}

static void fin_ticketpinned(TicketPinned t) {
    Pfree(t.tstarts);
    Pfree(t.offsets);
}
}

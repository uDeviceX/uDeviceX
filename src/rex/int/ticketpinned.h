namespace rex {
static void ini_ticketpinned(TicketPinned *t) {
    Palloc(&t->starts, 27);
    Palloc(&t->counts, 26);
}

static void fin_ticketpinned(TicketPinned t) {
    Pfree(t.starts);
    Pfree(t.counts);
}
}

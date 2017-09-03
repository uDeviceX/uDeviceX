namespace x {
static void ini_ticketpinned(TicketPinned *t) {
    Palloc0(&t->starts, 27);
    Palloc0(&t->counts, 26);
}

static void fin_ticketpinned(TicketPinned t) {
    Pfree(t.starts);
    Pfree(t.counts);
}
}

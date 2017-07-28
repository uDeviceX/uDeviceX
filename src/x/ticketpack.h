namespace x {
static void ini_ticketpack(TicketPack *t) {
    t->tstarts = new DeviceBuffer<int>(27);
    t->tstarts_hst = new PinnedHostBuffer1<int>(27);
    t->offsets_hst = new PinnedHostBuffer1<int>(26);

    t->offsets = new DeviceBuffer<int>;
    Dalloc(&t->starts, 27* MAX_OBJ_TYPES);
    Dalloc(&t->counts, 26* MAX_OBJ_TYPES);
}

static void fin_ticketpack(TicketPack t) {
    delete t.tstarts;
    delete t.tstarts_hst;
    delete t.offsets_hst;

    delete t.starts;
    delete t.offsets;

    Dfree(t.counts);
}
}

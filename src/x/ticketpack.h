namespace x {
static void ini_ticketpack(TicketPack *t) {
    t->tstarts = new DeviceBuffer<int>(27);
    t->tstarts_hst = new PinnedHostBuffer1<int>(27);
    t->offsets_hst = new PinnedHostBuffer1<int>(26);

    t->counts = new DeviceBuffer<int>;
    t->starts = new DeviceBuffer<int>;
    t->offsets = new DeviceBuffer<int>;

    Dalloc(&t->counts0, 26*MAX_OBJ_TYPES);
}

static void fin_ticketpack(TicketPack t) {
    delete t.tstarts;
    delete t.tstarts_hst;
    delete t.offsets_hst;


    delete t.counts;
    delete t.starts;
    delete t.offsets;
}
}

namespace x {
static void ini_ticketpack(TicketPack *t) {
    t->tstarts = new DeviceBuffer<int>(27);
    t->tstarts_hst = new PinnedHostBuffer4<int>(27);
    t->tcounts_hst = new PinnedHostBuffer4<int>(26);

    t->counts = new DeviceBuffer<int>;
    t->starts = new DeviceBuffer<int>;
    t->offsets = new DeviceBuffer<int>;

}

static void fin_ticketpack(TicketPack t) {
    delete t.tstarts;
    delete t.tstarts_hst;
    delete t.tcounts_hst;

    delete t.counts;
    delete t.starts;
    delete t.offsets;
}
}

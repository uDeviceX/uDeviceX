namespace x {
static void ini_tickpack(TicketPack *t) {
    t->packstotalstart = new DeviceBuffer<int>(27);
    t->host_packstotalstart = new PinnedHostBuffer1<int>(27);
    t->host_packstotalcount = new PinnedHostBuffer1<int>(26);

    t->packscount = new DeviceBuffer<int>;
    t->packsstart = new DeviceBuffer<int>;
    t->packsoffset = new DeviceBuffer<int>;

}

static void fin_tickpack(TicketPack *t) {
    delete t->packstotalstart;
    delete t->host_packstotalstart;
    delete t->host_packstotalcount;

    delete t->packscount;
    delete t->packsstart;
    delete t->packsoffset;
}
}

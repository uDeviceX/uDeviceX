void ini(int maxp, /**/ BulkData **bd) {
    BulkData *b = new BulkData;
    Dalloc(&b->zipped_pp, 2 * maxp);
    Dalloc(&b->zipped_rr,     maxp);
    b->rnd = new rnd::KISS(0, 0, 0, 0);
    b->colors = NULL;
    *bd = b;
}

void ini(/**/ HaloData **h) {
    *h = new HaloData;
}

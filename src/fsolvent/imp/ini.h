void ini(int maxp, /**/ BulkData *b) {
    Dalloc(&b->zipped_pp, 2 * maxp);
    Dalloc(&b->zipped_rr,     maxp);
    b->rnd = new rnd::KISS(0, 0, 0, 0);
}

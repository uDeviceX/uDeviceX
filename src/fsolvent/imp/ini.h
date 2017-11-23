void ini(int maxp, /**/ BulkData *b) {
    Dalloc(&b->zipped_pp, 2 * maxp);
    Dalloc(&b->zipped_rr,     maxp);
}

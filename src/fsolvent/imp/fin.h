void fin(/**/ BulkData *b) {
    CC(d::Free(b->zipped_pp));
    CC(d::Free(b->zipped_rr));
}

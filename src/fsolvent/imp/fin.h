void fin(/**/ BulkData *b) {
    CC(d::Free(b->zipped_pp));
    CC(d::Free(b->zipped_rr));
    delete b->rnd;
    delete b;
}

void fin(/**/ HaloData *h) {
    for (int i = 0; i < 26; ++i)
        delete h->trunks[i];
    delete h;
}

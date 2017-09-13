static void free_pinned_counts(int **hc, int **dc) {
    if (*hc) CC(d::FreeHost(*hc));
    *hc = *dc = NULL;
}

void fin_pinned(/**/ hBags *hb, dBags *db) {
    for (int i = 0; i < NBAGS; ++i) {
        if (hb->data[i]) CC(d::FreeHost(hb->data[i]));
        db->data[i] = hb->data[i] = NULL;
    }
    free_pinned_counts(&hb->counts, &db->counts);
}

void fin(/**/ hBags *hb, dBags *db) {
    for (int i = 0; i < NBAGS; ++i) {
        if (hb->data[i]) free(hb->data[i]);
        if (db->data[i]) CC(d::Free(db->data[i]));
        db->data[i] = hb->data[i] = NULL;
    }
    free_pinned_counts(&hb->counts, &db->counts);
}

void fin(/**/ Stamp *s) {
    MC(m::Comm_free(&s->cart));
}

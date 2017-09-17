static void free_counts(int **hc) {
    free(*hc); *hc = NULL;
}

static void free_pair(int i, AllocMod mod, /**/ hBags *hb, dBags *db) {
    switch (mod) {
    case HST_ONLY:
        free(hb->data[i]);
        break;
    case DEV_ONLY:
        CC(d::Free(db->data[i]));
        break;
    case PINNED:
        CC(d::FreeHost(hb->data[i]));
        break;
    case NONE:
    default:
        break;
    }
}

void fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db) {
    /* fragments */
    for (int i = 0; i < NFRAGS; ++i)
        free_pair(i, fmod, /**/ hb, db);

    /* bulk */
    free_pair(frag_bulk, bmod, /**/ hb, db);

    free_counts(/**/ &hb->counts);
}

void fin_pinned(/**/ hBags *hb, dBags *db) {
    for (int i = 0; i < NBAGS; ++i) {
        if (hb->data[i]) CC(d::FreeHost(hb->data[i]));
        db->data[i] = hb->data[i] = NULL;
    }
    free_counts(&hb->counts);
}

void fin(/**/ hBags *hb, dBags *db) {
    for (int i = 0; i < NBAGS; ++i) {
        if (hb->data[i]) free(hb->data[i]);
        if (db->data[i]) CC(d::Free(db->data[i]));
        db->data[i] = hb->data[i] = NULL;
    }
    free_counts(&hb->counts);
}

void fin(/**/ hBags *hb) {
   for (int i = 0; i < NBAGS; ++i) {
        if (hb->data[i]) free(hb->data[i]);
        hb->data[i] = NULL;
    }
   free_counts(&hb->counts);
}

void fin(/**/ Stamp *s) {
    MC(m::Comm_free(&s->cart));
}

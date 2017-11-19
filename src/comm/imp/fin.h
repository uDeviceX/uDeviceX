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
    case PINNED_DEV:
        CC(d::FreeHost(hb->data[i]));
        CC(d::Free(db->data[i]));
        break;
    case NONE:
    default:
        break;
    }
}

int fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db) {
    /* fragments */
    for (int i = 0; i < NFRAGS; ++i)
        free_pair(i, fmod, /**/ hb, db);

    /* bulk */
    free_pair(frag_bulk, bmod, /**/ hb, db);
    free_counts(/**/ &hb->counts);
    return 0;
}

int fin(/**/ Stamp *s) {
    COMM_MC(m::Comm_free(&s->cart));
    return 0;
}

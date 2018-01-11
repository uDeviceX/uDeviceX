static void free_counts(int **hc) {
    UC(efree(*hc));
    *hc = NULL;
}

static void free_pair(int i, AllocMod mod, /**/ hBags *hb, dBags *db) {
    switch (mod) {
    case HST_ONLY:
        if (hb->data[i])
            UC(efree(hb->data[i]));
        break;
    case DEV_ONLY:
        if (db->data[i])
            CC(d::Free(db->data[i]));
        break;
    case PINNED_HST:
    case PINNED:
        if (hb->data[i])
            CC(d::FreeHost(hb->data[i]));
        break;
    case PINNED_DEV:
        if (hb->data[i])
            CC(d::FreeHost(hb->data[i]));
        if (db->data[i])
            CC(d::Free(db->data[i]));
        break;
    case NONE:
    default:
        break;
    }
}

int bags_fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db) {
    /* fragments */
    for (int i = 0; i < NFRAGS; ++i)
        free_pair(i, fmod, /**/ hb, db);

    /* bulk */
    free_pair(frag_bulk, bmod, /**/ hb, db);
    free_counts(/**/ &hb->counts);
    return 0;
}

int fin(/**/ Comm *c) {
    MC(m::Comm_free(&c->cart));
    return 0;
}

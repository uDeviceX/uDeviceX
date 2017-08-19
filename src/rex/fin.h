namespace rex {
static void fin_local() {
    int i;
    LocalHalo *h;
    for (i = 0; i < 26; i++) {
        h = local[i];
        Dfree(h->indexes);
        delete h->ff;
    }
}

static void fin_remote() {
    int i;
    RemoteHalo* h;
    for (i = 0; i < 26; i++) {
        h = remote[i];
        Dfree(h->dstate);
        Pfree(h->hstate);
        free(h->pp);
        Pfree(h->ff_pi);
        delete h;
    }
}

void fin() {
    fin_local();
    fin_remote();
}
}

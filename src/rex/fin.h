namespace rex {
static void fin_local() {
    int i;
    for (i = 0; i < 26; i++) {
        Dfree(local[i]->indexes);
        delete local[i]->ff;
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

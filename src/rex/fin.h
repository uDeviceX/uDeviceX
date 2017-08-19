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
    RemoteHalo* r;
    for (i = 0; i < 26; i++) {
        r = remote[i];
        Dfree(r->dstate);
        Pfree(r->hstate);
        delete r;
    }
}

void fin() {
    fin_local();
    fin_remote();
}
}

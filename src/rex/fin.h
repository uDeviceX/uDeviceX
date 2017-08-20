namespace rex {
static void fin_local() {
    int i;
    LocalHalo *h;
    for (i = 0; i < 26; i++) {
        h = &local[i];
        Dfree(h->indexes);
        Pfree(h->ff_pi);
    }
}

static void fin_remote() {
    int i;
    RemoteHalo* h;
    for (i = 0; i < 26; i++) {
        h = &remote[i];
        Pfree(h->pp_pi);
        Pfree(h->ff_pi);
    }
}

void fin() {
    fin_local();
    fin_remote();
}
}

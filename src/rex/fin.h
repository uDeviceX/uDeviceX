namespace rex {
static void fin_local() {
    int i;
    LFrag *h;
    for (i = 0; i < 26; i++) {
        h = &local[i];
        Dfree(h->indexes);
        Pfree(h->ff_pi);
    }
}

void fin() {
    fin_local();
}
}

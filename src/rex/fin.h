namespace rex {
static void fin_local() {
    int i;
    for (i = 0; i < 26; i++) delete local[i];
}

static void fin_remote() {
    int i;
    for (i = 0; i < 26; i++) delete remote[i];
}

void fin() {
    fin_local();
    fin_remote();
}
}

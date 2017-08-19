namespace rex {
void local_resize() {
    int i, n;
    LocalHalo *l;
    for (i = 0; i < 26; ++i) {
        n = send_counts[i];
        l = local[i];
        l->ff->resize(n);
    }
}

void resizeR() {
    int i;
    for (i = 0; i < 26; ++i) remote[i]->n = recv_counts[i];
}
}

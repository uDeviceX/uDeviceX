namespace rex {
void local_resize() {
    int i;
    LocalHalo *l;
    for (i = 0; i < 26; ++i) {
        l = local[i];
        l->n = send_counts[i];
    }
}

void resizeR() {
    int i;
    for (i = 0; i < 26; ++i) remote[i]->n = recv_counts[i];
}
}

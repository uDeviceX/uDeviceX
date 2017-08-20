namespace rex {
void local_resize() {
    int i;
    LocalHalo *l;
    for (i = 0; i < 26; ++i) {
        l = &local[i];
        l->n = send_counts[i];
    }
}

}

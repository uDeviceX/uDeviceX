namespace rex {
void local_resize() {
    int i;
    for (i = 0; i < 26; ++i) lo::resize(local[i], send_counts[i]);
}

}

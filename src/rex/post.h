namespace rex {
bool post_check() {
    bool packingfailed;
    int i;
    packingfailed = false;
    for (i = 0; i < 26; ++i) packingfailed |= send_counts[i] > lo::size(local[i]);
    return packingfailed;
}

void post_resize() {
    int *indexes[26];
    int i;
    for (i = 0; i < 26; ++i) indexes[i] = local[i]->indexes->D;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::indexes, indexes, sizeof(indexes), 0, H2D));
}

void local_resize() {
    int i;
    for (i = 0; i < 26; ++i) lo::resize(local[i], send_counts[i]);
}

}

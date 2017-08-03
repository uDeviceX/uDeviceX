namespace rex {
bool post_check() {
    bool packingfailed;
    int i;
    packingfailed = false;
    for (i = 0; i < 26; ++i) packingfailed |= send_counts[i] > local[i]->capacity();
    return packingfailed;
}

void post_resize() {
    int capacities[26];
    int *scattered_indices[26];
    int i;
    for (i = 0; i < 26; ++i) capacities[i] = local[i]->capacity();
    CC(cudaMemcpyToSymbolAsync(k_rex::g::capacities, capacities, sizeof(capacities), 0, H2D));
    for (i = 0; i < 26; ++i) scattered_indices[i] = local[i]->scattered_indices->D;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::scattered_indices, scattered_indices, sizeof(scattered_indices), 0, H2D));
}

void local_resize() {
    int i;
    for (i = 0; i < 26; ++i) local[i]->resize(send_counts[i]);
}

}

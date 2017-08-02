namespace rex {
void ini() {
    int i;
    mpDeviceMalloc(&packbuf);
    Palloc(&host_packbuf, MAX_PART_NUM);

    for (i = 0; i < 26; i++) local[i] = new LocalHalo;
    for (i = 0; i < 26; i++) remote[i] = new RemoteHalo;
        
    for (i = 0; i < 26; ++i) {
        int estimate = 10;
        remote[i]->preserve_resize(estimate);
        local[i]->resize(estimate);
        local[i]->update();

        CC(cudaMemcpyToSymbol(k_rex::g::capacities,
                              &local[i]->scattered_indices->C, sizeof(int),
                              sizeof(int) * i, H2D));
        CC(cudaMemcpyToSymbol(k_rex::g::scattered_indices,
                              &local[i]->scattered_indices->D, sizeof(int *),
                              sizeof(int *) * i, H2D));
    }
}
}

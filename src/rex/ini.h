namespace rex {
void adjust_packbuffers() {
    int s = 0;
    for (int i = 0; i < 26; ++i) s += 32 * ((local[i]->capacity() + 31) / 32);
    packbuf->resize(s);
    host_packbuf->resize(s);
}

void ini() {
    int i;
    packbuf = new DeviceBuffer<Particle>;
    host_packbuf = new PinnedHostBuffer<Particle>;

    for (i = 0; i < 26; i++) local[i] = new LocalHalo;
    for (i = 0; i < 26; i++) remote[i] = new RemoteHalo;
        
    for (i = 0; i < 26; ++i) {
        int estimate = 10;
        remote[i]->preserve_resize(estimate);
        local[i]->resize(estimate);
        local[i]->update();

        CC(cudaMemcpyToSymbol(k_rex::g::ccapacities,
                              &local[i]->scattered_indices->C, sizeof(int),
                              sizeof(int) * i, H2D));
        CC(cudaMemcpyToSymbol(k_rex::g::scattered_indices,
                              &local[i]->scattered_indices->D, sizeof(int *),
                              sizeof(int *) * i, H2D));
    }

    adjust_packbuffers();

}
}

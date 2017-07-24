namespace rex {
void _adjust_packbuffers() {
    int s = 0;
    for (int i = 0; i < 26; ++i) s += 32 * ((local[i]->capacity() + 31) / 32);
    packbuf->resize(s);
    host_packbuf->resize(s);
}

void ini(/*io*/ basetags::TagGen *tg) {
    iterationcount = -1;
    packstotalstart = new DeviceBuffer<int>(27);
    host_packstotalstart = new PinnedHostBuffer<int>(27);
    host_packstotalcount = new PinnedHostBuffer<int>(26);

    packscount = new DeviceBuffer<int>;
    packsstart = new DeviceBuffer<int>;
    packsoffset = new DeviceBuffer<int>;

    packbuf = new DeviceBuffer<Particle>;
    host_packbuf = new PinnedHostBuffer<Particle>;

    for (int i = 0; i < SE_HALO_SIZE; i++) local[i] = new LocalHalo;
    for (int i = 0; i < SE_HALO_SIZE; i++) remote[i] = new RemoteHalo;

    btc  = get_tag(tg);
    btp1 = get_tag(tg);
    btp2 = get_tag(tg);
    btf  = get_tag(tg);
        
    for (int i = 0; i < 26; ++i) {
        int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
        recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
        int estimate = 1;
        remote[i]->preserve_resize(estimate);
        local[i]->resize(estimate);
        local[i]->update();

        CC(cudaMemcpyToSymbol(k_rex::ccapacities,
                              &local[i]->scattered_indices->C, sizeof(int),
                              sizeof(int) * i, H2D));
        CC(cudaMemcpyToSymbol(k_rex::scattered_indices,
                              &local[i]->scattered_indices->D, sizeof(int *),
                              sizeof(int *) * i, H2D));
    }

    _adjust_packbuffers();

}
}

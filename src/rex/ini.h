namespace rex {
static void ini_local(int estimate) {
    int i;
    for (i = 0; i < 26; i++) {
        local[i] = new LocalHalo;
        local[i]->indexes = new DeviceBuffer<int>;
        local[i]->ff      = new PinnedHostBuffer<Force>;
        local[i]->resize(estimate);
        local[i]->update();
    }
}

static void ini_remote(int estimate) {
    int i;
    for (i = 0; i < 26; i++) {
        remote[i] = new RemoteHalo;
        remote[i]->resize(estimate);
    }
}
static void ini_copy() {
    int i;
    for (i = 0; i < 26; ++i) {
        CC(cudaMemcpyToSymbol(k_rex::g::sizes,   &local[i]->indexes->C, sizeof(int),  sizeof(int)  * i, H2D));
        CC(cudaMemcpyToSymbol(k_rex::g::indexes, &local[i]->indexes->D, sizeof(int*), sizeof(int*) * i, H2D));
    }
}

void ini() {
    int estimate;
    estimate = 10;
    ini_local(estimate);
    ini_remote(estimate);
    ini_copy();
}
}

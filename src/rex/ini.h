namespace rex {
static void ini_local(int estimate) { }
static void ini_remote(int estimate) { }
static void ini_copy() { }

void ini() {
    int i, estimate;
    estimate = 10;
    ini_local(estimate);
    ini_remote(estimate);
    ini_copy();

    for (i = 0; i < 26; i++) local[i] = new LocalHalo;
    for (i = 0; i < 26; i++) remote[i] = new RemoteHalo;

    for (i = 0; i < 26; ++i) {
        remote[i]->resize(estimate);
        local[i]->resize(estimate);
        local[i]->update();

        CC(cudaMemcpyToSymbol(k_rex::g::sizes,   &local[i]->indexes->C, sizeof(int),  sizeof(int)  * i, H2D));
        CC(cudaMemcpyToSymbol(k_rex::g::indexes, &local[i]->indexes->D, sizeof(int*), sizeof(int*) * i, H2D));
    }
}
}

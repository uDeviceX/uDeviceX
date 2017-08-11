namespace rex {
static void i2d(int i, /**/ int d[3]) { /* fragment id to directiron */
    enum {X, Y, Z};
    d[X] = (i     + 2) % 3 - 1;
    d[Y] = (i / 3 + 2) % 3 - 1;
    d[Z] = (i / 9 + 2) % 3 - 1;
}

static int d2sz(int d[3]) { /* direction to size */
    enum {X, Y, Z};
    int x, y, z;
    x = (d[X] == 0 ? XS : 1);
    y = (d[Y] == 0 ? YS : 1);
    z = (d[Z] == 0 ? ZS : 1);
    return x * y * z;
}

static int i2sz(int i) { /* fragment id to size */
    int d[3];
    i2d(i, d);
    return d2sz(d);
}

static void ini_local(int estimate) {
    int i;
    for (i = 0; i < 26; i++) {
        local[i] = new LocalHalo;
        local[i]->indexes = new DeviceBuffer<int>;
        local[i]->ff      = new PinnedHostBuffer<Force>;
        lo::resize(local[i], estimate);
        lo::update(local[i]);
    }
}

static void ini_remote(int estimate) {
    int i;
    for (i = 0; i < 26; i++) {
        remote[i] = new RemoteHalo;
        re::resize(remote[i], estimate);
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

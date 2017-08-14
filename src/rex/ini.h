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

static int i2max(int i) { /* fragment id to maximum size */
    return MAX_OBJ_DENSITY*i2sz(i);
}

static void ini_local() {
    int i, n;
    for (i = 0; i < 26; i++) {
        n = i2max(i);
        local[i] = new LocalHalo;
        Dalloc(&local[i]->indexes, n);
        local[i]->ff      = new PinnedHostBuffer<Force>(n);
        lo::update(local[i]);
    }
}

static void ini_remote() {
    int i, n;
    for (i = 0; i < 26; i++) {
        n = i2max(i);
        remote[i] = new RemoteHalo;
        re::resize(remote[i], n);
        remote[i]->h.update(n);
    }
}
static void ini_copy() {
    int i;
    for (i = 0; i < 26; ++i)
        CC(cudaMemcpyToSymbol(k_rex::g::indexes, &local[i]->indexes, sizeof(int*), sizeof(int*) * i, H2D));
}

void ini() {
    ini_local();
    ini_remote();
    ini_copy();
}
}

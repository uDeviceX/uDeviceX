
static int scan(const int n, const int *counts, int27 *starts) {
    int i, s;
    starts->d[0] = 0;
    for (i = 0, s = 0; i < n; ++i)
        starts->d[i + 1] = (s += counts[i]);
    return s;
}

int unpack(const hBags bags, /**/ Particle *pp) {
    enum {NFRAGS=26};
    int nhalo, s, i;
    size_t c, bs = bags.bsize;
    int27 starts;

    nhalo = scan(NFRAGS, bags.counts, &starts);

    for (i = 0; i < NFRAGS; ++i) {
        c = bags.counts[i] * bs;
        s = starts.d[i];
        CC(d::MemcpyAsync(pp + s, bags.data[i], c, H2D));
    }
    
    return nhalo;
}

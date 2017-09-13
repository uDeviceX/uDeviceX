
static int scan(const int n, const int *counts, int27 *starts) {
    int i, s;
    starts->d[0] = 0;
    for (i = 0, s = 0; i < n; ++i)
        starts->d[i + 1] = (s += counts[i]);
    return s;
}

template <typename T>
static void unpack(const hBags bags, int27 starts, /**/ T *buf) {
    int s, i;
    size_t c, bs = bags.bsize;

    assert(bs == sizeof(T));
    
    for (i = 0; i < NFRAGS; ++i) {
        c = bags.counts[i] * bs;
        s = starts.d[i];
        CC(d::MemcpyAsync(buf + s, bags.data[i], c, H2D));
    }
}

int unpack_pp(const hBags bags, /**/ Particle *pp) {
    int nhalo;
    int27 starts;

    nhalo = scan(NFRAGS, bags.counts, &starts);
    
    unpack(bags, starts, /**/ pp);
    
    return nhalo;
}

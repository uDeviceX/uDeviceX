static void reini_map(Map m) {
    CC(d::MemsetAsync(m.counts, 0, 26 * sizeof(int)));
}

void build_map(int n, const Particle *pp, Map m) {
    reini_map(/**/ m);
    KL(dev::build_map, (k_cnf(n)), (pp, n, /**/ m));
    KL(dev::scan_map, (1, 32), (/**/ m));    
}

typedef Sarray <float2*, 26> float2p26;

template <typename T>
static void bag2Sarray(dBags bags, Sarray<T*, 26> *buf) {
    for (int i = 0; i < 26; ++i)
        buf->d[i] = (T*) bags.data[i];
}

void pack(const Map m, const Particle *pp, int n, /**/ dBags bags) {

    const int S = sizeof(Particle) / sizeof(float2);
    float2p26 wrap;
    bag2Sarray(bags, &wrap);

    KL((dev::pack<float2, S>), (k_cnf(S*n)), ((const float2*)pp, m, /**/ wrap));
}

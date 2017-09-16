template <typename T>
static void bag2Sarray(dBags bags, Sarray<T*, NFRAGS> *buf) {
    for (int i = 0; i < NFRAGS; ++i)
        buf->d[i] = (T*) bags.data[i];
}

static void pack_pp(const Map m, const Particle *pp, int n, /**/ dBags bags) {

    const int S = sizeof(Particle) / sizeof(float2);
    float2p26 wrap;
    bag2Sarray(bags, &wrap);

    KL((dev::pack<float2, S>), (k_cnf(S*n)), ((const float2*)pp, m, /**/ wrap));
}

static void pack_ii(const Map m, const int *ii, int n, /**/ dBags bags) {

    const int S = 1;
    intp26 wrap;
    bag2Sarray(bags, &wrap);

    KL((dev::pack<int, S>), (k_cnf(S*n)), (ii, m, /**/ wrap));
}

void pack_pp(const Particle *pp, int n, /**/ Pack *p) {
    pack_pp(p->map, pp, n, /**/ p->dpp);
}

void pack_ii(const int *ii, int n, /**/ Pack *p) {
    pack_ii(p->map, ii, n, /**/ p->dii);
}

void pack_cc(const int *cc, int n, /**/ Pack *p) {
    pack_ii(p->map, cc, n, /**/ p->dcc);
}

void download(int n, Pack *p) {
    CC(d::Memcpy(p->hpp.counts, p->map.counts, 26 * sizeof(int), D2H));
    if (global_ids)    CC(d::Memcpy(p->hii.counts, p->map.counts, 26 * sizeof(int), D2H));
    if (multi_solvent) CC(d::Memcpy(p->hcc.counts, p->map.counts, 26 * sizeof(int), D2H));

    int nhalo, i, c;
    for (i = nhalo = 0; i < NFRAGS; ++i) {
        c = p->hpp.counts[i];
        nhalo += c;
    }
    p->nbulk = n - nhalo;
}

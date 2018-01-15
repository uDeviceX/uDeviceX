static int reduce(int n, const int d[]) {
    int s, i;
    for (i = s = 0; i < n; ++i) s += d[i];
    return s;
}

static void pack_pp(const DMap m, const Particle *pp, /**/ dBags bags) {
    int n;
    const int S = sizeof(Particle) / sizeof(float2);
    float2p26 wrap;
    bag2Sarray(bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dev::pack<float2, S>), (k_cnf(S*n)), ((const float2*)pp, m, /**/ wrap));
}

static void pack_ii(const DMap m, const int *ii, /**/ dBags bags) {
    int n;
    const int S = 1;
    intp26 wrap;
    bag2Sarray(bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dev::pack<int, S>), (k_cnf(S*n)), (ii, m, /**/ wrap));
}

void dflu_pack(const FluQuants *q, /**/ DFluPack *p) {
    pack_pp(p->map, q->pp, /**/ p->dpp);
    if (global_ids)    pack_ii(p->map, q->ii, /**/ p->dii);
    if (multi_solvent) pack_ii(p->map, q->cc, /**/ p->dcc);
}

static void check_counts(int nfrags, const int *counts, const hBags *hpp) {
    enum {X, Y, Z};
    int i, c, cap;
    
    for (i = 0; i < nfrags; ++i) {
        c = counts[i];
        cap = comm_get_number_capacity(i, hpp);
        int f[3] = frag_i2d3(i);
        if (c > cap)
            ERR("exceed capacity in frag %d = [%d %d %d] : %d / %d",
                i, f[X], f[Y], f[Z], c, cap);
    }
}

void dflu_download(DFluPack *p) {
    const size_t sz = NFRAGS * sizeof(int);
    const int *counts = p->map.hcounts;
    check_counts(NFRAGS, counts, &p->hpp);

    dSync(); /* wait for pack kernels */
    memcpy(p->hpp.counts, counts, sz);
    if (global_ids)    memcpy(p->hii.counts, counts, sz);
    if (multi_solvent) memcpy(p->hcc.counts, counts, sz);

    p->nhalo = reduce(NFRAGS, counts);
}

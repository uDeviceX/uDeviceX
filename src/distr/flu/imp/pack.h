static int reduce(int n, const int d[]) {
    int s, i;
    for (i = s = 0; i < n; ++i) s += d[i];
    return s;
}

static void pack_pp(const Map m, const Particle *pp, /**/ dBags bags) {
    int n;
    const int S = sizeof(Particle) / sizeof(float2);
    float2p26 wrap;
    bag2Sarray(bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dev::pack<float2, S>), (k_cnf(S*n)), ((const float2*)pp, m, /**/ wrap));
}

static void pack_ii(const Map m, const int *ii, /**/ dBags bags) {
    int n;
    const int S = 1;
    intp26 wrap;
    bag2Sarray(bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dev::pack<int, S>), (k_cnf(S*n)), (ii, m, /**/ wrap));
}

void pack(const Quants *q, /**/ Pack *p) {
    pack_pp(p->map, q->pp, /**/ p->dpp);
    if (global_ids)    pack_ii(p->map, q->ii, /**/ p->dii);
    if (multi_solvent) pack_ii(p->map, q->cc, /**/ p->dcc);
}

void download(int n, Pack *p) {
    dSync(); /* wait for pack kernels */
    const size_t sz = NFRAGS * sizeof(int);
    memcpy(p->hpp.counts, p->map.hcounts, sz);
    if (global_ids)    memcpy(p->hii.counts, p->map.hcounts, sz);
    if (multi_solvent) memcpy(p->hcc.counts, p->map.hcounts, sz);

    int nhalo, i, c;
    for (i = nhalo = 0; i < NFRAGS; ++i) {
        c = p->hpp.counts[i];
        nhalo += c;
    }
    p->nbulk = n - nhalo;
}

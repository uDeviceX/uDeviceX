static void pack_mesh(int nv, const Particle *pp, Map map, /**/ Pap26 buf) {
    KL(dev::pack_mesh, (14 * 16, 128), (nv, pp, map, /**/ buf));
}

void pack(int nv, const Particle *pp, /**/ Pack *p) {
    Pap26 wrap;
    bag2Sarray(p->dpp, &wrap);
    pack_mesh(nv, pp, p->map, /**/ wrap);
}

void download(Pack *p) {
    download_counts(1, p->map, /**/ p->hpp.counts);
}


static int27 scan(const int cc[26]) {
    int i, s;
    int27 ss;
    s = ss.d[0] = 0;
    for (i = 0; i < 26; ++i) ss.d[i+1] = (s += cc[i]);
    return ss;
}

static void pack_mom(const int counts[], const Momentum *mm, /**/ Mop26 buf) {
    int27 starts = scan(counts);
    int n = starts.d[26];
    KL(dev::pack_mom, (k_cnf(n)), (starts, mm, /**/ buf));
}

void packM(const int counts[], const Momentum *mm, /**/ PackM *p) {
    Mop26 wrap;
    bag2Sarray(p->dmm, &wrap);
    pack_mom(counts, mm, /**/ wrap);
}

void download(PackM *p) {
    dSync(); // wait for pack_mom to complete
}


template <typename T>
static void bag2Sarray(dBags bags, Sarray<T*, NBAGS> *buf) {
    for (int i = 0; i < NBAGS; ++i)
        buf->d[i] = (T*) bags.data[i];
}

static void pack_pp(const Map m, int ns, int nv, const Particle *ipp, /**/ dBags bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);

    enum {THR=128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(nv, THR), ns);
        
    KL((dev::pack_pp_packets), (blck, thrd), (nv, ipp, m, /**/ wrap));
}

static void pack_ss(const Map m, int n, const Solid *ss, /**/ dBags bags) {

    Sarray<Solid*, 27> wrap;
    bag2Sarray(bags, &wrap);

    KL((dev::pack_ss), (k_cnf(n)), (ss, m, /**/ wrap));
}


void pack(int ns, int nv, const Solid *ss, const Particle *ipp, /**/ Pack *p) {
    pack_pp(p->map, ns, nv, ipp, /**/ p->dipp);
    pack_ss(p->map, ns, ss, /**/ p->dss);
}

void download(Pack *p) {
    CC(d::Memcpy(p->hipp.counts, p->map.counts, NBAGS * sizeof(int), D2H));
    CC(d::Memcpy(p->hss.counts, p->map.counts, NBAGS * sizeof(int), D2H));
}

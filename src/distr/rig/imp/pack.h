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

void pack_pp(int ns, int nv, const Particle *ipp, /**/ Pack *p) {
    pack_pp(p->map, ns, nv, ipp, /**/ p->dipp);
}

void download(Pack *p) {
    CC(d::Memcpy(p->hipp.counts, p->map.counts, NBAGS * sizeof(int), D2H));
    CC(d::Memcpy(p->hss.counts, p->map.counts, NBAGS * sizeof(int), D2H));
}

template <typename T>
static void bag2Sarray(dBags bags, Sarray<T*, NBAGS> *buf) {
    for (int i = 0; i < NBAGS; ++i)
        buf->d[i] = (T*) bags.data[i];
}

static void pack_pp(const Map m, int nc, int nv, const Particle *pp, /**/ dBags bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);

    enum {THR=128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(nv, THR), nc);
        
    KL((dev::pack_pp), (blck, thrd), (nv, pp, m, /**/ wrap));
}

void pack_pp(int nc, int nv, const Particle *pp, /**/ Pack *p) {
    pack_pp(p->map, nc, nv,  pp, /**/ p->dpp);
}

void download(int nc, Pack *p) {
    int *cc = p->hpp.counts;
    CC(d::Memcpy(p->hpp.counts, p->map.counts, NBAGS * sizeof(int), D2H));
    p->nbulk = cc[frag_bulk];
}

// TODO this should be in common place
template <typename T>
static void bag2Sarray(dBags bags, Sarray<T*, NFRAGS> *buf) {
    for (int i = 0; i < NFRAGS; ++i)
        buf->d[i] = (T*) bags.data[i];
}

static void pack_pp(int nfrags, int nw, const ParticlesWrap *ww, Map map, /**/ Particlep26 buf) {
    int i, stride;
    stride = nfrags + 1;
    const ParticlesWrap *w;
    PackHelper ph;
    
    for (i = 0; i < nw; ++i) {
        w = ww + i;
        ph.starts  = map.starts  + i * stride;
        ph.offsets = map.offsets + i * stride;
        memcpy(ph.indices, map.ids, nfrags * sizeof(int*));

        KL(dev::pack_pp, (14 * 16, 128), (w->pp, ph, /**/ buf));
    }
}

void pack(int nw, const ParticlesWrap *ww, Map map, /**/ Pack *p) {
    Particlep26 wrap;
    bag2Sarray(p->dpp, &wrap);
    pack_pp(NFRAGS, nw, ww, p->map, /**/ wrap);
}

void download(Pack *p) {
    CC(d::Memcpy(p->hpp.counts, p->map.counts, NFRAGS * sizeof(int), D2H));    
}

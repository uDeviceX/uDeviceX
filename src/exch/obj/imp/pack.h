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

void download(int nw, Pack *p) {
    int *src, *dst;
    size_t sz = NBAGS * sizeof(int);
    src = p->map.offsets + nw * NBAGS;
    dst = p->hpp.counts;
    CC(d::Memcpy(dst, src, sz, D2H));    
}

static void clear_forces(int nfrags, /**/ PackF *p) {
    int i, c;
    size_t sz;
    for (i = 0; i < nfrags; ++i) {
        c = p->hff.counts[i];
        sz = c * p->hff.bsize;
        if (c) CC(d::MemsetAsync(p->dff.data[i], 0, sz));
    }   
}

void reini_ff(const Pack *p, PackF *pf) {
    size_t sz = NBAGS * sizeof(int);
    memcpy(pf->hff.counts, p->hpp.counts, sz);
    clear_forces(NFRAGS, /**/ pf);
}

void download_ff(PackF *p) {
    int i, c;
    size_t sz;
    data_t *src, *dst;
    for (i = 0; i < NFRAGS; ++i) {
        c = p->hff.counts[i];
        sz = c * p->hff.bsize;
        src = p->dff.data[i];
        dst = p->hff.data[i];
        if (c) CC(d::MemcpyAsync(dst, src, sz, D2H));
    }
    dSync();
}

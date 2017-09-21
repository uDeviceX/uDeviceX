void upload(Unpack *u) {
    int i, c;
    size_t sz;
    data_t *src, *dst;

    for (i = 0; i < NFRAGS; ++i) {
        c = u->hpp.counts[i];
        if (c) {
            sz  = u->hpp.bsize * c;
            dst = u->dpp.data[i];
            src = u->hpp.data[i];
            CC(d::MemcpyAsync(dst, src, sz, H2D));
        }
    }
}

static void unpack_ff(int nfrags, Forcep26 ff, Map map, int nw, /**/ ForcesWrap *ww) {
    int i, stride;
    stride = nfrags + 1;
    const ForcesWrap *w;
    PackHelper ph;
    
    for (i = 0; i < nw; ++i) {
        w = ww + i;
        ph.starts  = map.starts  + i * stride;
        ph.offsets = map.offsets + i * stride;
        memcpy(ph.indices, map.ids, nfrags * sizeof(int*));

        KL(dev::unpack_ff, (14 * 16, 128), (ff, ph, /**/ w->ff));
    }
}

void unpack_ff(const UnpackF *u, const Pack *p, int nw, /**/ ForcesWrap *ww) {
    Forcep26 wrap;
    bag2Sarray(u->dff, &wrap);
    unpack_ff(NFRAGS, wrap, p->map, nw, /**/ ww);
}

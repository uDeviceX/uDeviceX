

void pack_pp(int nfrags, int nw, const ParticlesWrap *ww, Map map, /**/ Particlep26 buf) {
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

static void pack_pp(int nfrags, int n, const Particle *pp, Map map, /**/ Pap26 buf) {
    PackHelper ph;
    
    ph.starts  = map.starts;
    ph.offsets = map.offsets;
    memcpy(ph.indices, map.ids, nfrags * sizeof(int*));
    
    KL(dev::pack_pp, (14 * 16, 128), (pp, ph, /**/ buf));
}

void pack(int n, const Particle *pp, /**/ Pack *p) {
    Pap26 wrap;
    bag2Sarray(p->dpp, &wrap);
    pack_pp(NFRAGS, n, pp, p->map, /**/ wrap);
}

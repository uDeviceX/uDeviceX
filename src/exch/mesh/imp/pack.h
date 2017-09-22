static void pack_mesh(int nv, const Particle *pp, Map map, /**/ Pap26 buf) {
    KL(dev::pack_mesh, (14 * 16, 128), (nv, pp, map, /**/ buf));
}

void pack(int nv, const Particle *pp, /**/ Pack *p) {
    Pap26 wrap;
    bag2Sarray(p->dpp, &wrap);
    pack_mesh(nv, pp, p->map, /**/ wrap);
}

void download(Pack *p) {
    int *src, *dst;
    size_t sz = NBAGS * sizeof(int);
    src = p->map.offsets + NBAGS;
    dst = p->hpp.counts;
    CC(d::Memcpy(dst, src, sz, D2H));
}


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


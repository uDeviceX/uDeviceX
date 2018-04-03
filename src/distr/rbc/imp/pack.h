static void pack_pp(const DMap m, int nc, int nv, const Particle *pp, /**/ dBags bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);
    UC(dcommon_pack_pp_packets(nc, nv, pp, m, /**/ wrap));
}

/* all data (including map) on host */
static void pack_ii(const DMap m, int nc, const int *ii, /**/ hBags bags) {
    Sarray<int*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);

    int i, c, j, src;
    for (i = 0; i < NBAGS; ++i) {
        c = m.counts[i];
        for (j = 0; j < c; ++j) {
            src = m.ids[i][j];
            wrap.d[i][j] = ii[src];
        }
    }
}

void drbc_pack(const RbcQuants *q, /**/ DRbcPack *p) {
    UC(pack_pp(p->map, q->nc, q->nv, q->pp, /**/ p->dpp));
    if (p->ids) {
        UC(dmap_D2H(NBAGS, &p->map, /**/ &p->hmap));
        dSync();
        UC(pack_ii(p->hmap, q->nc, q->ii, /**/ p->hii));
    }
}

void drbc_download(DRbcPack *p) {
    size_t sz = NBAGS * sizeof(int);
    CC(d::Memcpy(p->hpp.counts, p->map.counts, sz, D2H));
    if (p->ids)
        memcpy(p->hii.counts, p->hmap.counts, sz);
}

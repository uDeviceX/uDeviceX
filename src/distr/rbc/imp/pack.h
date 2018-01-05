static void pack_pp(const Map m, int nc, int nv, const Particle *pp, /**/ dBags bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);

    enum {THR=128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(nv, THR), nc);
        
    KL((dev::pack_pp_packets), (blck, thrd), (nv, pp, m, /**/ wrap));
}

/* all data (including map) on host */
static void pack_ii(const Map m, int nc, const int *ii, /**/ hBags bags) {
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

void pack(const rbc::Quants *q, /**/ Pack *p) {
    pack_pp(p->map, q->nc, q->nv, q->pp, /**/ p->dpp);

    if (rbc_ids) {
        map_D2H(NBAGS, &p->map, /**/ &p->hmap);
        dSync();
        pack_ii(p->hmap, q->nc, q->ii, /**/ p->hii);
    }
}

void download(Pack *p) {
    size_t sz = NBAGS * sizeof(int);
    CC(d::Memcpy(p->hpp.counts, p->map.counts, sz, D2H));
    if (rbc_ids)
        memcpy(p->hii.counts, p->hmap.counts, sz);
}

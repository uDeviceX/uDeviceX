static void pack_pp(const Map m, int nc, int nv, const Particle *pp, /**/ dBags bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);

    enum {THR=128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(nv, THR), nc);
        
    KL((dev::pack_pp_packets), (blck, thrd), (nv, pp, m, /**/ wrap));
}

/* all data (including map) on host */
static void pack_ii(const Map m, int nc, const int *ii, /**/ dBags bags) {
    Sarray<int*, NBAGS> wrap;
    bag2Sarray(bags, &wrap);

    int i, c, j, src;
    for (i = 0; i < NBAGS; ++i) {
        c = m.counts[j];
        for (j = 0; j < c; ++j) {
            src = m.ids[i][j];
            wrap.d[i][j] = ii[src];
        }
    }
}

void pack(int nc, int nv, const Particle *pp, /**/ Pack *p) {
    pack_pp(p->map, nc, nv,  pp, /**/ p->dpp);

    // if (rbc_ids)
    //     pack_ii(p->hmap, nc, ii, /**/ p->hii);
}

void download(Pack *p) {
    CC(d::Memcpy(p->hpp.counts, p->map.counts, NBAGS * sizeof(int), D2H));
}

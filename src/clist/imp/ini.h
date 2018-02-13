void clist_ini(int LX, int LY, int LZ, /**/ Clist *c) {
    c->dims.x = LX;
    c->dims.y = LY;
    c->dims.z = LZ;
    c->ncells = LX * LY * LZ;

    size_t size = (c->ncells + 16) * sizeof(int);
    CC(d::Malloc((void **) &c->starts, size));
    CC(d::Malloc((void **) &c->counts, size));
}

void clist_ini_map(int maxp, int nA, const Clist *c, /**/ ClistMap **map) {
    size_t size;
    EMALLOC(1, map);
    ClistMap *m = *map;
    
    UC(scan_ini(c->ncells, /**/ &m->scan));

    size = maxp * sizeof(uchar4);

    m->maxp = maxp;
    
    m->nA = nA;
    if (nA > MAXA)
        ERR("Too many inputs (%d / %d)", nA, MAXA);
    
    for (int i = 0; i < nA; ++i)
        CC(d::Malloc((void **) &m->ee[i], size));

    size = nA * maxp * sizeof(uint);
    CC(d::Malloc((void **) &m->ii, size));
}

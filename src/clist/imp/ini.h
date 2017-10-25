void ini(int LX, int LY, int LZ, /**/ Clist *c) {
    c->dims.x = LX;
    c->dims.y = LY;
    c->dims.z = LZ;
    c->ncells = LX * LY * LZ;

    size_t size = (c->ncells + 1) * sizeof(int);
    CC(d::Malloc((void **) &c->starts, size));
    CC(d::Malloc((void **) &c->counts, size));
    
}

void ini_map(int nA, const Clist *c, /**/ Map *m) {
    size_t size;
    scan::alloc_work(c->ncells, /**/ &m->scan);

    size = MAX_PART_NUM * sizeof(uchar4);

    m->nA = nA;
    assert(nA <= MAXA);
    
    for (int i = 0; i < nA; ++i)
        CC(d::Malloc((void **) &m->ee[i], size));

    size = MAX_PART_NUM * sizeof(uint);
    CC(d::Malloc((void **) &m->ii, size));
}

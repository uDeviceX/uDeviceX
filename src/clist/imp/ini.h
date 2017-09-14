void ini(int LX, int LY, int LZ, /**/ Clist *c) {
    c->dims.x = LX;
    c->dims.y = LY;
    c->dims.z = LZ;
    c->ncells = LX * LY * LZ;

    size_t size = (c->ncells + 1) * sizeof(int);
    CC(d::Malloc((void **) &c->starts, size));
    CC(d::Malloc((void **) &c->counts, size));
    
}

void ini_ticket(const Clist *c, /**/ Ticket *t) {
    size_t size;
    scan::alloc_work(c->ncells, /**/ &t->scan);

    size = MAX_PART_NUM * sizeof(uchar4);
    CC(d::Malloc((void **) &t->eelo, size));
    CC(d::Malloc((void **) &t->eere, size));

    size = MAX_PART_NUM * sizeof(uint);
    CC(d::Malloc((void **) &t->ii, size));
}

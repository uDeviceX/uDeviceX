void ini(int LX, int LY, int LZ, /**/ Clist *c) {
    c->dims.x = LX;
    c->dims.y = LY;
    c->dims.z = LZ;
    c->ncells = LX * LY * LZ;

    size_t size = (c->ncells + 1) * sizeof(int);
    CC(d::Malloc((void **) &c->starts, size));
    CC(d::Malloc((void **) &c->counts, size));
    
}
void ini_work(Work *w) {

}

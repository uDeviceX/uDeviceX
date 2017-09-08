static void estimates(int nfrags, float maxd, /**/ int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = frag_ncell(i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}

static void alloc_one_pinned_frag(int i, /**/ Bags *b) {
    size_t n = b->bsize * b->capacity[i];
    CC(d::alloc_pinned(&b->hst[i], n));
    CC(d::HostGetDevicePointer(&b->dev[i], b->hst[i], 0));
}

static void ini_bags(int nfrags, size_t bsize, float maxdensity, /**/ Bags *b) {
    b->bsize = bsize;
    estimates(nfrags, maxdensity, b->capacity);
    for (int i = 0; i < nfrags; ++i) alloc_one_pinned_frag(i, /**/ b);
}

void ini_no_bulk(size_t bsize, float maxdensity, /**/ Bags *b) {
    ini_bags(NFRAGS, bsize, maxdensity, /**/ b);
    b->hst = b->dev = NULL;
}

void ini_full(size_t bsize, float maxdensity, /**/ Bags *b) {
    ini_bags(NBAGS, bsize, maxdensity, /**/ b);
}



void ini(MPI_Comm comm, /**/ Stamp *s) {

}

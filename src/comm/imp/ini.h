
static void estimates(int nfrags, float maxd, /**/ int *counts) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = frag_ncell(i);
        e = (int) (e * maxd);
        counts[i] = e;
    }
}

static void alloc_one_frag(int i, /**/ Bags *b) {
    size_t n = b->bsize * b->counts[i];
    CC(d::alloc_pinned(&b->hst[i], n));
    CC(d::HostGetDevicePointer(&b->dev[i], b->hst[i], 0));
}

void ini_full(size_t bsize, int cap[NBAGS], float maxdensity, /**/ Bags *b) {
    b->bsize = bsize;
    estimates(NFRAGS, maxdensity, b->counts);

    
}

void ini(MPI_Comm comm, /**/ Stamp *s) {

}

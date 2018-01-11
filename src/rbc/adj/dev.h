namespace adj {
static __device__ int fetch(const int *a, int i) { return a[i]; }
static __device__ int dev(int md, int nv, int i, const int *adj0, const int *adj1, /**/ AdjMap *m) {
    int i0, i1, i2, i3, i4;
    int rbc, offset, j, k;

    assert(md == RBCmd);
    assert(nv == RBCnv);

    i0 = i / md;
    j  = i % md;

    k   = (i0 % nv) * md;
    i1 = fetch(adj0, k + j);
    if (i1 == -1) return 0; /* invalid */

    i2 = fetch(adj0, k + ((j + 1) % md));
    if (i2 == -1) {
        i2 = fetch(adj0, k    );
        i3 = fetch(adj0, k + 1);
    } else {
        i3 = fetch(adj0, k + ((j + 2) % md));
        if (i3 == -1) i3 = fetch(adj0, k);
    }
    i4 = fetch(adj1, k + j);

    rbc = i0 / nv;
    offset = rbc * nv;
    i1 += offset; i2 += offset; i3 += offset; i4 += offset; /* no i0 */

    assert(rbc < MAX_CELL_NUM);

    m->rbc = rbc;
    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    return 1;
}
} /* namespace */

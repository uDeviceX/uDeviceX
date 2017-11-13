/* helps to find indices for triangles and dihedrals */

struct Map {
    int valid; /* == 0 if not valid */
    int i0, i1, i2, i3, i4;
};

static __device__ void ini_map(int md, int nv, int i, const Texo<int> adj0, const Texo<int> adj1, /**/ Map *m) {
    int i0, i1, i2, i3, i4;
    int valid;
    int offset, j, k;

    assert(md == RBCmd);
    assert(nv == RBCnv);

    i0 = i / md;
    j  = i % md;

    k   = (i0 % nv) * md;
    i1 = fetch(adj0, k + j);
    if (i1 == -1) {
        m->valid = 0;
        return ;
    }

    i2 = fetch(adj0, k + ((j + 1) % md));
    if (i2 == -1) {
        i2 = fetch(adj0, k    );
        i3 = fetch(adj0, k + 1);
    } else {
        i3 = fetch(adj0, k + ((j + 2) % md));
        if (i3 == -1) i3 = fetch(adj0, k);
    }
    i4 = fetch(adj1, k + j);

    offset  = (i0 / nv) * nv;
    i1 += offset;
    i2 += offset;
    i3 += offset;
    i4 += offset;

    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    m->valid = 1;
}

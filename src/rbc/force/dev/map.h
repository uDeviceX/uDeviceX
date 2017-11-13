/* helps to find indices for triangles and dihedrals */

struct Map {
    int valid; /* == 0 if not valid */
    int i0, i1, i2, i3, i4;
};

static __device__ void ini_map(int md, int nv, int i, const Texo<int> adj0, const Texo<int> adj1, /**/ Map *m) {
    int i0, i1, i2, i3, i4;
    int valid;
    int lid, idrbc, offset, neighid;
    
    assert(md == RBCmd);
    assert(nv == RBCnv);

    i0 =      i / md;
    neighid = i % md;
    
    lid   = i0 % nv;
    idrbc = i0 / nv;
    
    offset = idrbc * nv;
    i1 = fetch(adj0, neighid + md * lid);
    valid = i1 != -1;

    i2 = fetch(adj0, ((neighid + 1) % md) + md * lid);
    if (i2 == -1 && valid) {
        i2 = fetch(adj0, 0 + md * lid);
        i3 = fetch(adj0, 1 + md * lid);
    } else {
        i3 = fetch(adj0, ((neighid + 2) % md) + md * lid);
        if (i3 == -1 && valid) i3 = fetch(adj0, 0 + md * lid);
    }
    i4 = fetch(adj1, neighid + md * lid);

    i1 += offset;
    i2 += offset;
    i3 += offset;
    i4 += offset;

    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    m->valid = valid;
}

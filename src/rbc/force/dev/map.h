/* helps to find indices for triangles and dihedrals */

struct Map {
    int tri, dih; /* == 0 if not valid */
    int i0, i1, i2, i3, i4;
};

void ini_map(int md, int nv, int i, const Texo<int> adj0, const Texo<int> adj1, /**/ Map *m) {
    assert(md == RBCmd);
    assert(nv == RBCnv);
}

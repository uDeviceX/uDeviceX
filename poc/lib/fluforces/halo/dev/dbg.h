template <typename Parray>
static __device__ void assert_frag(int3 L, int i, const RFrag_v<Parray> frag) {
    int xs, ys, zs; /* sizes */
    int dx, dy, dz;
    xs = frag.xcells; ys = frag.ycells; zs = frag.zcells;
    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    assert(xs * ys * zs == frag_dev::ncell(L, i));
    assert(frag_dev::d2i(dx, dy, dz) == i);
}

template <typename Parray>
static __device__ void assert_rc(int3 L, const RFrag_v<Parray> frag, int i, int row, int col, int jump) {
    /* i: base cell id */
    int fid, nmax;
    int dx, dy, dz;
    int j;
    assert(row == 1 || row == 2 || row == 3);
    assert(col == 1 || col == 2 || col == 3);

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    assert(dx == -1 || dx == 0 || dx == 1);
    assert(dy == -1 || dy == 0 || dy == 1);
    assert(dz == -1 || dz == 0 || dz == 1);
    fid  = frag_dev::d2i(dx, dy, dz);
    nmax = frag_dev::ncell(L, fid) + 1;

    for (j = 0 ; j < row; j++) {
        assert(i       < nmax);
        assert(i + col < nmax);
        i += jump;
    }
}

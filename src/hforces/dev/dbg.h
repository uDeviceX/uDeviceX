__device__ void assert_frag(int i, const Frag frag) {
    int xs, ys, zs; /* sizes */
    int dx, dy, dz;
    xs = frag.xcells; ys = frag.ycells; zs = frag.zcells;
    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    assert(xs * ys * zs == frag_ncell(i));
    assert(frag_d2i(dx, dy, dz) == i);
}

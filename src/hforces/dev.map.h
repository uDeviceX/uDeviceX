namespace hforces { namespace dev {

struct Map { /* helps to find remote particle */
    int  org0, org1, org2;
    int cnt0, cnt1, cnt2;
};

/* true if `i' bigger than the number of remote particles */
static __device__ int endp(const Map m, int i) { return i >= m.cnt2; }

static __device__ int m2id(const Map m, int i) {
    /* return remote particle id */
    int m1, m2;
    int id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}

static __device__ int get(const int *a) { return *a; }
static __device__ Map r2map0(const Frag frag,
                             int basecid, int row, int col, int ncols) {
    int org0, org1, org2;
    int cnt0, cnt1, cnt2;
    int count1, count2;
    const int* start;
    Map m;

    start = frag.start + basecid;
    org0 = get(start);
    cnt0 = get(start + col) - org0;
    start += ncols;

    org1   = org2 = 0;
    count1 = count2 = 0;
    if (row > 1) {
        org1   = get(start);
        count1 = get(start + col) - org1;
        start += ncols;
    }
    if (row > 2) {
        org2   = get(start);
        count2 = get(start + col) - org2;
    }
    cnt1 = cnt0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m.org0 = org0; m.org1 = org1; m.org2 = org2;
    m.cnt0 = cnt0; m.cnt1 = cnt1; m.cnt2 = cnt2;
    return m;
}

static __device__ void xyz2rc(int type,
                              int dx, int dy, int dz, /* fragment information */
                              int xc, int yc, int zc,
                              int xs, int ys, int zs, /* size */
                              /**/ int *prow, int *pcol, int *pncols) {
    int row, col, ncols;
    if         (type == FACE) {
        row = dz ? ys : zs;
        col = dx ? ys : xs;
        ncols = dx ? yc : xc;
    } else if (type == EDGE)
        col = max(xs, max(ys, zs));
    else if (type == CORNER) {
        row = col = ncols = 1;
    } else {
        printf("%s:%d: illigal fragmant type: %d\n", __FILE__, __LINE__, type);
        assert(0);
    }
    *prow = row; *pcol = col; *pncols = ncols;
}

static __device__  void r2size(int r, int nc, int S, /**/ int *pl, int *ps) {
    int i, s, l;
    i = (int)(r + S/2);
    l = max(0,  i - 1);
    s = min(nc, i + 2) - l;
    *pl = l; *ps = s;
}
static __device__ Map r2map(const Frag frag, float x, float y, float z) {
    /* coordinate [r] to map */
    int id;
    int xl, yl, zl; /* low */
    int xs, ys, zs; /* size */
    int dx, dy, dz;
    int xc, yc, zc;
    int row, col, ncols;

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    xc = frag.xcells; yc = frag.ycells; zc = frag.zcells;

    id = 0; xs = ys = zs = 1;
    if (dz == 0) {r2size(z, zc, ZS, /**/ &zl, &zs); id += zl;}
    id *= yc;
    if (dy == 0) {r2size(y, yc, YS, /**/ &yl, &ys); id += yl;}
    id *= xc;
    if (dx == 0) {r2size(x, xc, XS, /**/ &xl, &xs); id += xl;}
    xyz2rc(frag.type,
           dx, dy, dz,
           xc, yc, zc,
           xs, ys, zs,
           &row, &col, &ncols);
    return r2map0(frag, id, row, col, ncols);
}

} } /* namespace */

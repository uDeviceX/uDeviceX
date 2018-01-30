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

static __device__ int get(const int *a, int i) { return a[i]; }
static __device__ Map r2map0(const int *start, int i, int row, int col, int jump) {
    int j;
    int o[3] = {0}, c[3] = {0};
    Map m;
    assert(row == 1 || row == 2 || row == 3);
    assert(col == 1 || col == 2 || col == 3);

    for (j = 0 ; j < row; j++) {
        o[j] = get(start, i);
        c[j] = get(start, i + col) - o[j];
        i += jump;
    }
    c[1] += c[0]; c[2] += c[1];
    o[1] -= c[0]; o[2] -= c[1];

    m.org0 = o[0]; m.org1 = o[1]; m.org2 = o[2];
    m.cnt0 = c[0]; m.cnt1 = c[1]; m.cnt2 = c[2];
    return m;
}


static __device__ int facep(int dx, int dy, int dz) { /* face? */
    int x, y, z;
    x =  dx  && !dy && !dz;
    y =  !dx &&  dy && !dz;
    z =  !dx && !dy &&  dz;
    return x || y || z;
}
static __device__ void xyz2rc_face(int dx, int dy, int dz,
                                   int xc, int yc, int zc,
                                   int xs, int ys, int zs,
                                   /**/ int *prow, int *pcol, int *pjump) {
    int row, col, jump;
    if (!facep(dx, dy, dz)) {
        printf("xyz: %d %d %d\n", dx, dy, dz);
        assert(0);
    }
    if (dx) {
        row   = zs;
        col   = ys;
        jump = yc;
    } else if (dy) {
        row   = zs;
        col   = xs;
        jump = xc;
    } else { /* (dz) */
        row   = ys;
        col   = xs;
        jump = xc;
    }
    *prow = row; *pcol = col; *pjump = jump;
}

static __device__ void xyz2rc(int type,
                              int dx, int dy, int dz, /* fragment information */
                              int xc, int yc, int zc,
                              int xs, int ys, int zs, /* size */
                              /**/ int *prow, int *pcol, int *pjump) {
    int row, col, jump;
    if      (type == flu::FACE)
        xyz2rc_face(dx, dy, dz, xc, yc, zc, xs, ys, zs, /**/ &row, &col, &jump);
    else if (type == flu::EDGE) {
        row = jump = 1;
        col = max(xs, max(ys, zs));
    }
    else if (type == flu::CORNER) {
        row = col = jump = 1;
    } else {
        printf("%s:%d: illigal fragmant type: %d [%d %d %d]\n",
               __FILE__, __LINE__, type, dx, dy, dz);
        assert(0);
    }
    *prow = row; *pcol = col; *pjump = jump;
}

static __device__  void r2size(float r, int nc, int S, /**/ int *pl, int *ps) {
    int i, s, l;
    i = (int)(r + S/2);
    l = max(0,  i - 1);
    s = min(nc, i + 2) - l;
    *pl = l; *ps = s;
}

static __device__ Map r2map(int3 L, const flu::RFrag frag, float x, float y, float z) {
    /* coordinate [r] to map */
    int id; /* base id */
    int xl, yl, zl; /* low */
    int xs, ys, zs; /* size */
    int dx, dy, dz;
    int xc, yc, zc;
    int row, col, jump;

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    xc = frag.xcells; yc = frag.ycells; zc = frag.zcells;

    id = 0; xs = 1; ys = 1; zs = 1;
    if (dz == 0) {r2size(z, zc, L.z, /**/ &zl, &zs); id += zl;}
    id *= yc;
    if (dy == 0) {r2size(y, yc, L.y, /**/ &yl, &ys); id += yl;}
    id *= xc;
    if (dx == 0) {r2size(x, xc, L.x, /**/ &xl, &xs); id += xl;}

    xyz2rc(frag.type,
           dx, dy, dz,
           xc, yc, zc,
           xs, ys, zs,
           &row, &col, &jump);
    assert_rc(frag, id, row, col, jump);
    return r2map0(frag.start, id, row, col, jump);
}

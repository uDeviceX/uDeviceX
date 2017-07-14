namespace k_bipsbatch {
struct Map { /* helps to find remote particle */
    int  org0, org1, org2;
    uint cnt0, cnt1, cnt2;
};

/* true if `i' bigger than the number of remote particles */
static __device__ int endp(const Map m, uint i) { return i >= m.cnt2; }

static __device__ uint m2id(const Map m, uint i) {
    /* return remote particle id */
    int m1, m2;
    uint id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}

static __device__ Map p2map(const Frag frag, float x, float y, float z) {
    /* create map for a particle */
    int org0, org1, org2;
    uint cnt0, cnt1, cnt2;
    int count1, count2;
    int basecid;
    int xcid, ycid, zcid;
    int xl, yl, zl; /* low */
    int xs, ys, zs; /* size */
    int dx, dy, dz;
    int row, col, ncols;
    int* start;
    Map m;

    dx = frag.dx; dy = frag.dy; dz = frag.dz;

    basecid = 0; xs = 1; ys = 1; zs = 1;
    if (dz == 0) {
        zcid = (int)(z + ZS / 2);
        zl = max(0, -1 + zcid);
        zs = min(frag.zcells, zcid + 2) - zl;
        basecid = zl;
    }
    basecid *= frag.ycells;

    if (dy == 0) {
        ycid = (int)(y + YS / 2);
        yl = max(0, -1 + ycid);
        ys = min(frag.ycells, ycid + 2) - yl;
        basecid += yl;
    }
    basecid *= frag.xcells;

    if (dx == 0) {
        xcid = (int)(x + XS / 2);
        xl = max(0, -1 + xcid);
        xs = min(frag.xcells, xcid + 2) - xl;
        basecid += xl;
    }

    row = col = ncols = 1;
    if (frag.type == FACE) {
        row = dz ? ys : zs;
        col = dx ? ys : xs;
        ncols = dx ? frag.ycells : frag.xcells;
    } else if (frag.type == EDGE)
        col = max(xs, max(ys, zs));

    start = frag.start + basecid;
    org0 = __ldg(start);
    cnt0 = __ldg(start + col) - org0;
    start += ncols;

    org1   = org2 = 0;
    count1 = count2 = 0;
    if (row > 1) {
        org1   = __ldg(start);
        count1 = __ldg(start + col) - org1;
        start += ncols;
    }
    if (row > 2) {
        org2   = __ldg(start);
        count2 = __ldg(start + col) - org2;
    }
    cnt1 = cnt0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m.org0 = org0; m.org1 = org1; m.org2 = org2;
    m.cnt0 = cnt0; m.cnt1 = cnt1; m.cnt2 = cnt2;
    return m;
}
}

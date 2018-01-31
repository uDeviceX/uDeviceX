static __device__ int minmax(int lo, int hi, int a) {
    return min(hi, max(lo, a));
}

static __device__ void ini(int3 L, int zplane, const Texo<int> texstart, int w_n, float x, float y, float z, /**/ Map *m) {
#define start_fetch(i) (fetch(texstart, i))
#define   wpp_fetch(i) (fetch(texwpp,   i))
    uint cnt0, cnt1, cnt2, org0;
    int org1, org2;

    int xbase = (int)(x - (-XS / 2 - XWM));
    int ybase = (int)(y - (-YS / 2 - YWM));
    int zbase = (int)(z - (-ZS / 2 - ZWM));

    xbase = minmax(-XWM+1, XS + XWM - 2, xbase);
    ybase = minmax(-YWM+1, YS + YWM - 2, ybase);
    zbase = minmax(-ZWM+1, ZS + ZWM - 2, zbase);

    enum {
        XCELLS = XS + 2 * XWM,
        YCELLS = YS + 2 * YWM,
        ZCELLS = ZS + 2 * ZWM,
        NCELLS = XCELLS * YCELLS * ZCELLS
    };

    int cid0 = xbase - 1 + XCELLS * (ybase - 1 + YCELLS * (zbase - 1 + zplane));

    org0 = start_fetch(cid0);
    int count0 = start_fetch(cid0 + 3) - org0;

    int cid1 = cid0 + XCELLS;
    org1 = start_fetch(cid1);
    int count1 = start_fetch(cid1 + 3) - org1;

    int cid2 = cid0 + XCELLS * 2;
    org2 = start_fetch(cid2);
    int count2 = cid2 + 3 == NCELLS
        ? w_n
        : start_fetch(cid2 + 3) - org2;

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
#undef start_fetch
#undef wpp_fetch
}

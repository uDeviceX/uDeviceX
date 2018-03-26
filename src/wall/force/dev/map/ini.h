static __device__ int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }
static __device__ void ini(int3 L, int zplane, const Texo<int> texstart, int w_n, float x, float y, float z, /**/ Map *m) {
#define start_fetch(i) (fetch(texstart, i))
#define   wpp_fetch(i) (fetch(texwpp,   i))
    uint cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xcells, ycells, zcells, ncells;
    int xbase, ybase, zbase;    
    
    xbase = (int)(x + L.x / 2 + XWM);
    ybase = (int)(y + L.y / 2 + YWM);
    zbase = (int)(z + L.z / 2 + ZWM);

    xbase = minmax(-XWM+1, L.x + XWM - 2, xbase);
    ybase = minmax(-YWM+1, L.y + YWM - 2, ybase);
    zbase = minmax(-ZWM+1, L.z + ZWM - 2, zbase);

    xcells = L.x + 2 * XWM;
    ycells = L.y + 2 * YWM;
    zcells = L.z + 2 * ZWM;

    ncells = xcells * ycells * zcells;

    int cid0 = xbase - 1 + xcells * (ybase - 1 + ycells * (zbase - 1 + zplane));

    org0 = start_fetch(cid0);
    int count0 = start_fetch(cid0 + 3) - org0;

    int cid1 = cid0 + xcells;
    org1 = start_fetch(cid1);
    int count1 = start_fetch(cid1 + 3) - org1;

    int cid2 = cid0 + xcells * 2;
    org2 = start_fetch(cid2);
    int count2 = cid2 + 3 == ncells
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

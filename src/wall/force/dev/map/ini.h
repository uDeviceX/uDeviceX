static __device__ int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }

static __device__ void ini(int3 L, int zplane, const Texo<int> texstart, int w_n, float x, float y, float z, /**/ Map *m) {
#define start_fetch(i) (fetch(texstart, i))
    int str0, str1, str2;
    int org0, org1, org2;
    int xcells, ycells, zcells, ncells;
    int xbase, ybase, zbase;
    int cid0, cid1, cid2;
    int count0, count1, count2;

    xcells = L.x + 2 * XWM;
    ycells = L.y + 2 * YWM;
    zcells = L.z + 2 * ZWM;

    ncells = xcells * ycells * zcells;

    xbase = (int)(x + xcells / 2);
    ybase = (int)(y + ycells / 2);
    zbase = (int)(z + zcells / 2);

    xbase = minmax(1, xcells - 2, xbase);
    ybase = minmax(1, ycells - 2, ybase);
    zbase = minmax(1, zcells - 2, zbase);

    cid0 = xbase - 1 + xcells * (ybase - 1 + ycells * (zbase - 1 + zplane));
    org0 = start_fetch(cid0);
    count0 = start_fetch(cid0 + 3) - org0;

    cid1 = cid0 + xcells;
    org1 = start_fetch(cid1);
    count1 = start_fetch(cid1 + 3) - org1;

    cid2 = cid0 + xcells * 2;
    org2 = start_fetch(cid2);
    count2 = cid2 + 3 == ncells
        ? w_n
        : start_fetch(cid2 + 3) - org2;

    str0 = count0;
    str1 = count0 + count1;
    str2 = str1 + count2;

    org1 -= str0;
    org2 -= str1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->str0 = str0; m->str1 = str1; m->str2 = str2;
#undef start_fetch
}

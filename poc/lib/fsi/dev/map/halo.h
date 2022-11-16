static __device__ int tex2map(int3 L, const int *start, int zplane, int n1, float x, float y, float z, /**/ Map *m) {
    /* textures to map */
    int xstart, xcount;
    int zmy;
    int xcenter, ycenter, zcenter;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;
    int cnt0, cnt1, cnt2;
    int org0, org1, org2;
    int xoffset, yoffset, zoffset;
    int ncells;

    xoffset = L.x / 2;
    yoffset = L.y / 2;
    zoffset = L.z / 2;
    ncells = L.x * L.y * L.z;

    xcenter = xoffset + (int)floorf(x);
    xstart = max(0, xcenter - 1);
    xcount = min(L.x, xcenter + 2) - xstart;

    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return 0;

    ycenter = yoffset + (int)floorf(y);

    zcenter = zoffset + (int)floorf(z);
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < L.z;

    count0 = count1 = count2 = 0;

    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = start[cid0];
        count0 = ((cid0 + xcount == ncells)
                  ? n1
                  : start[cid0 + xcount]) -
            org0;
    }

    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = start[cid1];
        count1 = ((cid1 + xcount == ncells)
                  ? n1
                  : start[cid1 + xcount]) -
            org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = start[cid2];
        count2 = ((cid2 + xcount == ncells)
                  ? n1
                  : start[cid2 + xcount]) -
            org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return 1;
}

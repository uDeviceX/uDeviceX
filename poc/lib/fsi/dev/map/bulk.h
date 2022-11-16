static __device__ int r2map(int3 L, const int *start, int zplane, int n, float x, float y, float z, /**/ Map *m) {
    /* coordinate [r] to map */
    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xoffset, yoffset, zoffset;
    int ncells;

    xoffset = L.x / 2;
    yoffset = L.y / 2;
    zoffset = L.z / 2;
    ncells = L.x * L.y * L.z;

    const int xcenter = xoffset + (int)floorf(x);
    const int xstart = max(0, xcenter - 1);
    const int xcount = min(L.x, xcenter + 2) - xstart;
    
    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return 0;
        
    const int ycenter = yoffset + (int)floorf(y);
        
    const int zcenter = zoffset + (int)floorf(z);
    const int zmy = zcenter - 1 + zplane;
    const bool zvalid = zmy >= 0 && zmy < L.z;
    
    int count0 = 0, count1 = 0, count2 = 0;
    
    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        const int cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = start[cid0];
        count0 = ((cid0 + xcount == ncells)
                  ? n
                  : start[cid0 + xcount]) -
            org0;
    }
    
    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        const int cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = start[cid1];
        count1 = ((cid1 + xcount == ncells)
                  ? n
                  : start[cid1 + xcount]) -
            org1;
    }
    
    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        const int cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = start[cid2];
        count2 = ((cid2 + xcount == ncells)
                  ? n
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

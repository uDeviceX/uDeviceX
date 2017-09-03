static __device__ int r2map(int zplane, int n, float x, float y, float z, /**/ Map *m) {
    /* coordinate [r] to map */
    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    enum {
        XCELLS = XS,
        YCELLS = YS,
        ZCELLS = ZS,
        XOFFSET = XCELLS / 2,
        YOFFSET = YCELLS / 2,
        ZOFFSET = ZCELLS / 2,
        NCELLS  = XS*YS*ZS
    };

    const int xcenter = XOFFSET + (int)floorf(x);
    const int xstart = max(0, xcenter - 1);
    const int xcount = min(XCELLS, xcenter + 2) - xstart;
    
    if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return 0;
        
    const int ycenter = YOFFSET + (int)floorf(y);
        
    const int zcenter = ZOFFSET + (int)floorf(z);
    const int zmy = zcenter - 1 + zplane;
    const bool zvalid = zmy >= 0 && zmy < ZCELLS;
    
    int count0 = 0, count1 = 0, count2 = 0;
    
    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
        const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
        org0 = fetchS(cid0);
        count0 = ((cid0 + xcount == NCELLS)
                  ? n
                  : fetchS(cid0 + xcount)) -
            org0;
    }
    
    if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
        const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
        org1 = fetchS(cid1);
        count1 = ((cid1 + xcount == NCELLS)
                  ? n
                  : fetchS(cid1 + xcount)) -
            org1;
    }
    
    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
        const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
        org2 = fetchS(cid2);
        count2 = ((cid2 + xcount == NCELLS)
                  ? n
                  : fetchS(cid2 + xcount)) -
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

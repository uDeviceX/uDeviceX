namespace k_fsi {
static __device__ int tex2map(int zplane, int n1, float x, float y, float z, /**/ Map *m) {
    /* textures to map */
    int xstart, xcount;
    int zmy;
    int xcenter, ycenter, zcenter;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;
    int cnt0, cnt1, cnt2;
    int org0, org1, org2;
    
    enum {
        XCELLS = XS,
        YCELLS = YS,
        ZCELLS = ZS,
        XOFFSET = XCELLS / 2,
        YOFFSET = YCELLS / 2,
        ZOFFSET = ZCELLS / 2,
        NCELLS  = XS*YS*ZS
    };
    
    xcenter = XOFFSET + (int)floorf(x);
    xstart = max(0, xcenter - 1);
    xcount = min(XCELLS, xcenter + 2) - xstart;
    
    if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return 0;
    
    ycenter = YOFFSET + (int)floorf(y);
    
    zcenter = ZOFFSET + (int)floorf(z);
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < ZCELLS;
    
    count0 = count1 = count2 = 0;
    
    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
        cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
        org0 = tex1Dfetch(t::start, cid0);
        count0 = ((cid0 + xcount == NCELLS)
                  ? n1
                  : tex1Dfetch(t::start, cid0 + xcount)) -
            org0;
    }
    
    if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
        cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
        org1 = tex1Dfetch(t::start, cid1);
        count1 = ((cid1 + xcount == NCELLS)
                  ? n1
                  : tex1Dfetch(t::start, cid1 + xcount)) -
            org1;
    }
    
    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
        cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
        org2 = tex1Dfetch(t::start, cid2);
        count2 = ((cid2 + xcount == NCELLS)
                  ? n1
                  : tex1Dfetch(t::start, cid2 + xcount)) -
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
}

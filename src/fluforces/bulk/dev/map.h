#define _S_ static __device__
#define _I_ static __device__

struct Map {
    int org0, org1, org2, org3, org4;
    int str0, str1, str2, str3, str4;
};

_I_ int map_end(const Map *m) {
    return m->str4;
}

_I_ int map_get_id(const Map *m, int i) {
    int m1, m2, m3, m4, id;
    m1 = (i >= m->str0);
    m2 = (i >= m->str1);
    m3 = (i >= m->str2);
    m4 = (i >= m->str3);
    id = i + (m4 ? m->org4 :
              m3 ? m->org3 :
              m2 ? m->org2 :
              m1 ? m->org1 :
              m->org0);
    return id;
}

_S_ bool valid_c(int c, int hi) {
    return (c >= 0) && (c < hi);
}

_S_ int2 part_map_bounds(int3 L, int dz, int dy, int3 ca, const int *start) {
    int3 cb;
    int2 bounds;
    int enddx, startx, endx, cid0;
    cb.z = ca.z + dz;
    cb.y = ca.y + dy;
    if (!valid_c(cb.z, L.z)) return make_int2(0, 0);
    if (!valid_c(cb.y, L.y)) return make_int2(0, 0);

    /* dx runs from -1 to enddx */
    enddx = (dz == 0 && dy == 0) ? 0 : 1;

    startx =     max(    0, ca.x - 1    );
    endx   = 1 + min(L.x-1, ca.x + enddx);

    cid0 = L.x * (cb.y + L.y * cb.z);

    bounds.x = start[cid0 + startx];
    bounds.y = start[cid0 + endx];

    return bounds;
}

_I_ void map_build(int3 L, int3 ca, const int *start, Map *m) {
#define GET_BOUNDS(dz, dy) part_map_bounds (L, dz, dy, ca, start)
    int2 b;   /* bounds */
    int s; /* start, count */

    b = GET_BOUNDS(-1, -1);
    m->org0 = b.x;
    s = b.y - b.x;
    m->str0 = s;
    
    b = GET_BOUNDS(-1,  0);
    m->org1 = b.x - s;
    s += b.y - b.x;
    m->str1 = s;
    
    b = GET_BOUNDS(-1,  1);
    m->org2 = b.x - s;
    s += b.y - b.x;
    m->str2 = s;

    b = GET_BOUNDS( 0, -1);
    m->org3 = b.x - s;
    s += b.y - b.x;
    m->str3 = s;

    b = GET_BOUNDS( 0,  0);
    m->org4 = b.x - s;
    s += b.y - b.x;
    m->str4 = s;
    
#undef GET_BOUNDS    
}

#undef _S_
#undef _I_

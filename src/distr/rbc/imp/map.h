static void reini_map(int n, const float *rr, /**/ Map *map) {
    
}

static int get_fid(const float r[3]) {
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -XS/2) + (r[X] >= XS/2);
    y = -1 + (r[Y] >= -YS/2) + (r[Y] >= YS/2);
    z = -1 + (r[Z] >= -ZS/2) + (r[Z] >= ZS/2);
    return frag_d2i(x, y, z);
}

static void build_map(int n, const float *rr, /**/ Map *map) { 
    int i, fid, dst;
    for (i = 0; i < n; ++i) {
        fid = get_fid(rr + 3 * i);
        
        dst = map->counts[fid]++;
        map->ids[fid][dst] = i;
    }
}


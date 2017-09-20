namespace dev {

enum { MAX_DSTS = 7 };
enum {BULK, FACE, EDGE, CORN};
enum {X, Y, Z};

__device__ int map_code(int3 L, const float r[3]) {
    int x, y, z;
    
    x = -1 + (r[X] >= -L.x / 2) + (r[X] >= L.x / 2);
    y = -1 + (r[Y] >= -L.y / 2) + (r[Y] >= L.y / 2);
    z = -1 + (r[Z] >= -L.z / 2) + (r[Z] >= L.z / 2);

    return frag_d2i(x, y, z);
}

__device__ int add_faces(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    int c, code;
    for (int c = 0; c < 3; ++c) {
        if (d[c]) {
            int df[3] = {0, 0, 0}; df[c] = d[c];
            code = frag_d2i(df[X], df[Y], df[Z]);
            fids[j++] = code;
        }
    }
    return j;
}

__device__ int add_edges(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    int c, code;
    for (c = 0; c < 3; ++c) {
        int de[3] = {d[X], d[Y], d[Z]}; de[c] = 0;
        if (de[(c + 1) % 3] && de[(c + 2) % 3]) {
            code = frag_d2i(de[X], de[Y], de[Z]);
            fids[j++] = code;
        }
    }
    return j;
}

__device__ int add_cornr(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    if (d[X] && d[Y] && d[Z]) {
        int code = d2i(d[X], d[Y], d[Z]);
        fids[j++] = code;
    }
    return j;
}

__device__ int map_decode(int code, /**/ int fids[MAX_DSTS]) {
    int j = 0;
    const int d[3] = frag_i2d3(code);
    j = add_faces(j, d, /**/ dsts);
    j = add_edges(j, d, /**/ dsts);
    j = add_cornr(j, d, /**/ dsts);
    return j;
}


} // dev

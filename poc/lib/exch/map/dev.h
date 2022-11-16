#define _I_ static __device__
#define _S_ static __device__

/* corner fragments are neighbor with 7 fragments */
enum { MAX_DSTS = 7 };

// tag::int[]
_I_ int emap_code(int3 L, const float r[3]) // <1>
// end::int[]
{
    int x, y, z;
    enum {X, Y, Z};
    x = -1 + (r[X] >= -L.x / 2) + (r[X] >= L.x / 2);
    y = -1 + (r[Y] >= -L.y / 2) + (r[Y] >= L.y / 2);
    z = -1 + (r[Z] >= -L.z / 2) + (r[Z] >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

// tag::int[]
_I_ int emap_code_box(int3 L, float3 lo, float3 hi)  // <2>
// end::int[]
{
    int x, y, z;
    x = -1 + (lo.x >= -L.x / 2) + (hi.x >= L.x / 2);
    y = -1 + (lo.y >= -L.y / 2) + (hi.y >= L.y / 2);
    z = -1 + (lo.z >= -L.z / 2) + (hi.z >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

// tag::int[]
_I_ void emap_add(int nfrags, int soluteid, int pid, int fid, EMap m) // <3>
// end::int[]
{
    int ientry, centry, stride, cap;
    cap = m.cap[fid];
    stride = nfrags + 1;
    centry = soluteid * stride + fid;
    ientry = atomicAdd(m.counts + centry, 1);
    if (ientry < cap)
        m.ids[fid][ientry] = pid;
}

_S_ int add_faces(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    for (int c = 0; c < 3; ++c) {
        if (d[c]) {
            int df[3] = {0, 0, 0}; df[c] = d[c];
            fids[j++] = frag_dev::d32i(df);
        }
    }
    return j;
}

_S_ int add_edges(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    for (int c = 0; c < 3; ++c) {
        int de[3] = {d[X], d[Y], d[Z]}; de[c] = 0;
        if (de[(c + 1) % 3] && de[(c + 2) % 3])
            fids[j++] = frag_dev::d32i(de);
    }
    return j;
}

_S_ int add_cornr(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    if (d[X] && d[Y] && d[Z])
        fids[j++] = frag_dev::d32i(d);
    return j;
}

// tag::int[]
_I_ int emap_decode(int code, /**/ int fids[MAX_DSTS]) // <4>
// end::int[]
{
    int j = 0;
    int d[3];
    frag_dev::i2d3(code, d);
    j = add_faces(j, d, /**/ fids);
    j = add_edges(j, d, /**/ fids);
    j = add_cornr(j, d, /**/ fids);
    return j;
}

#undef _I_
#undef _S_

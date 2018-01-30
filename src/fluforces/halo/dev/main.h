struct Fo { float *x, *y, *z; }; /* force */

static __device__ void pair(const forces::Pa a, const forces::Pa b, float rnd,
                            /**/ float *fx, float *fy, float *fz) {
    forces::Fo f;
    forces::gen(a, b, rnd, /**/ &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

static __device__ float random(int aid, int bid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? aid : bid;
    a2 = mask ? bid : aid;
    return rnd::mean0var1uu(seed, a1, a2);
}

static __device__ void force0(const flu::RndFrag rnd, const flu::RFrag bfrag, const Map m, const forces::Pa a, int aid, /**/
                              float *fx, float *fy, float *fz) {
    forces::Pa b;
    int i;
    int bid; /* id of b */
    float x, y, z; /* pair force */

    *fx = *fy = *fz = 0;
    for (i = 0; !endp(m, i); i ++ ) {
        bid = m2id(m, i);
        cloud_get(bfrag.c, bid, /**/ &b);
        pair(a, b, random(aid, bid, rnd.seed, rnd.mask), &x, &y, &z);
        *fx += x; *fy += y; *fz += z;
    }
}

static __device__ void force1(const flu::RndFrag rnd, const flu::RFrag frag, const Map m, const forces::Pa p, int id, Fo f) {
    float x, y, z; /* force */
    force0(rnd, frag, m, p, id, /**/ &x, &y, &z);
    atomicAdd(f.x, x);
    atomicAdd(f.y, y);
    atomicAdd(f.z, z);
}

static __device__ void force2(int3 L, const flu::RFrag frag, const flu::RndFrag rnd, forces::Pa p, int id, /**/ Fo f) {
    float x, y, z;
    Map m;
    forces::p2r3(&p, /**/ &x, &y, &z);
    m = r2map(L, frag, x, y, z);
    forces::shift(-frag.dx * L.x,
                  -frag.dy * L.y,
                  -frag.dz * L.z, /**/ &p);
    force1(rnd, frag, m, p, id, /**/ f);
}

static __device__ Fo i2f(const int *ii, float *ff, int i) {
    /* local id and index to force */
    Fo f;
    ff += 3*ii[i];
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}

static __device__ Fo Lfrag2f(const flu::LFrag frag, float *ff, int i) {
    return i2f(frag.ii, ff, i);
}

static __device__ void force3(int3 L, const flu::LFrag afrag, const flu::RFrag bfrag, const flu::RndFrag rnd, int i, /**/ float *ff) {
    forces::Pa p;
    Fo f;
    cloud_get(afrag.c, i, &p);
    f = Lfrag2f(afrag, ff, i);
    force2(L, bfrag, rnd, p, i, f);
}

__global__ void force(int3 L, const int27 start, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    flu::RndFrag  rnd;
    flu::RFrag rfrag;
    flu::LFrag lfrag;
    int gid;
    int fid; /* fragment id */
    int i; /* particle id */

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= start.d[26]) return;
    fid = frag_get_fid(start.d, gid);
    i = gid - start.d[fid];
    lfrag = lfrags.d[fid];
    if (i >= lfrag.n) return;

    rfrag = rfrags.d[fid];
    assert_frag(fid, rfrag);

    rnd = rrnd.d[fid];
    force3(L, lfrag, rfrag, rnd, i, /**/ ff);
}

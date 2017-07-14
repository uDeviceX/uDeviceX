namespace k_bipsbatch {
static __constant__ unsigned int start[27];

static __constant__ SFrag        ssfrag[26];
static __constant__ Frag          ffrag[26];
static __constant__ Rnd            rrnd[26];

struct Part { /* local particle */
    float x, y, z;
    float vx, vy, vz;
    float *fx, *fy, *fz;
    uint id;
};

struct Map { /* helps to find remote particle */
    int org0, org1, org2;
    int cnt0, cnt1, cnt2;
};

__device__ void force0(const Rnd rnd, float2 *pp, const Map m, Part p) {
    float x, y, z;
    float vx, vy, vz;
    float *fx, *fy, *fz;
    uint dpid;

     x = p.x;   y = p.y;   z = p.z;
    vx = p.vx; vy = p.vy; vz = p.vz;
    fx = p.fx; fy = p.fy; fz = p.fz;
    dpid = p.id;
    
    int mask = rnd.mask;
    float seed = rnd.seed;
    float xforce = 0, yforce = 0, zforce = 0;
    for (uint i = threadIdx.x & 1; i < m.cnt2; i += 2) {
        int m1 = (int)(i >= m.cnt0);
        int m2 = (int)(i >= m.cnt1);
        uint spid = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);

        float2 s0 = __ldg(pp + 0 + spid * 3);
        float2 s1 = __ldg(pp + 1 + spid * 3);
        float2 s2 = __ldg(pp + 2 + spid * 3);

        uint arg1 = mask ? dpid : spid;
        uint arg2 = mask ? spid : dpid;
        float myrandnr = l::rnd::d::mean0var1uu(seed, arg1, arg2);
        float3 r1 = make_float3(x, y, z), r2 = make_float3(s0.x, s0.y, s1.x);
        float3 v1 = make_float3(vx, vy, vz), v2 = make_float3(s1.y, s2.x, s2.y);
        float3 strength = force(SOLVENT_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, myrandnr);

        xforce += strength.x;
        yforce += strength.y;
        zforce += strength.z;
    }
    atomicAdd(fx, xforce);
    atomicAdd(fy, yforce);
    atomicAdd(fz, zforce);
}

static __device__ void p2r(const Part p, /**/ float *x, float *y, float *z) {
    *x = p.x; *y = p.y; *z = p.z;
}

static __device__ Map p2map(const Frag frag, const Part p) {
    float x, y, z;
    int org0, org1, org2;
    uint cnt0, cnt1, cnt2;
    int count1, count2;
    int basecid;
    int xcid, ycid, zcid;
    int xl, yl, zl; /* low */
    int xs, ys, zs; /* size */
    int dx, dy, dz;
    int row, col, ncols;
    int* start;
    Map m;

    p2r(p, &x, &y, &z);
    dx = frag.dx; dy = frag.dy; dz = frag.dz;

    basecid = 0; xs = 1; ys = 1; zs = 1;
    if (dz == 0) {
        zcid = (int)(z + ZS / 2);
        zl = max(0, -1 + zcid);
        zs = min(frag.zcells, zcid + 2) - zl;
        basecid = zl;
    }
    basecid *= frag.ycells;

    if (dy == 0) {
        ycid = (int)(y + YS / 2);
        yl = max(0, -1 + ycid);
        ys = min(frag.ycells, ycid + 2) - yl;
        basecid += yl;
    }
    basecid *= frag.xcells;

    if (dx == 0) {
        xcid = (int)(x + XS / 2);
        xl = max(0, -1 + xcid);
        xs = min(frag.xcells, xcid + 2) - xl;
        basecid += xl;
    }

    row = col = ncols = 1;
    if (frag.type == FACE) {
        row = dz ? ys : zs;
        col = dx ? ys : xs;
        ncols = dx ? frag.ycells : frag.xcells;
    } else if (frag.type == EDGE)
        col = max(xs, max(ys, zs));

    start = frag.start + basecid;
    org0 = __ldg(start);
    cnt0 = __ldg(start + col) - org0;
    start += ncols;

    org1   = org2 = 0;
    count1 = count2 = 0;
    if (row > 1) {
        org1   = __ldg(start);
        count1 = __ldg(start + col) - org1;
        start += ncols;
    }
    if (row > 2) {
        org2   = __ldg(start);
        count2 = __ldg(start + col) - org2;
    }
    cnt1 = cnt0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;
    

    m.org0 = org0; m.org1 = org1; m.org2 = org2;
    m.cnt0 = cnt0; m.cnt1 = cnt1; m.cnt2 = cnt2;
    return m;
}

__device__ void force1(const Frag frag, const Rnd rnd, /**/ Part p) {
    int dx, dy, dz;
    Map m;
    m = p2map(frag, p);

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    p.x -= dx * XS;
    p.y -= dy * YS;
    p.z -= dz * ZS;
    force0(rnd, frag.pp, m, p);
}

static __device__ void i2f(const int *ii, float *f, uint i, /**/ float **fx, float **fy, float **fz) {
    f += 3*ii[i];
    *fx = f++; *fy = f++; *fz = f++;
}

static __device__ void p2rv(const float *p,
                            uint i,
                            float  *x, float  *y, float  *z,
                            float *vx, float *vy, float *vz) {
    p += 6*i;
     *x = *(p++);  *y = *(p++);  *z = *(p++);
    *vx = *(p++); *vy = *(p++); *vz = *(p++);
}

static __device__ Part sfrag2p(const SFrag sfrag, float *ff, uint i) {
    Part p;
    p2rv(sfrag.pp,     i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    i2f (sfrag.ii, ff, i, /**/ &p.fx, &p.fy, &p.fz);
    p.id = i;
    return p;
}

__device__ void force2(const SFrag sfrag, const Frag frag, const Rnd rnd, uint i, /**/ float *ff) {
    Part p;
    p = sfrag2p(sfrag, ff, i);
    force1(frag, rnd, p);
}

static __device__ unsigned int get_hid(const unsigned int a[], const unsigned int i) {
    /* where is `i' in sorted a[27]? */
    unsigned int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

__global__ void force(/**/ float *ff) {
    Frag frag;
    Rnd  rnd;
    SFrag sfrag;
    int gid;
    uint hid; /* halo id */
    uint i; /* particle id */

    gid = (threadIdx.x + blockDim.x * blockIdx.x) >> 1;
    if (gid >= start[26]) return;
    hid = get_hid(start, gid);
    i = gid - start[hid];
    sfrag = ssfrag[hid];
    if (i >= sfrag.n) return;

    frag = ffrag[hid];
    rnd = rrnd[hid];
    force2(sfrag, frag, rnd, i, /**/ ff);
}
}

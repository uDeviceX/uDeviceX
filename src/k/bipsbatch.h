namespace k_bipsbatch {
static __constant__ unsigned int start[27];

static __constant__ SFrag        ssfrag[26];
static __constant__ Frag          ffrag[26];
static __constant__ Rnd            rrnd[26];

struct Part { /* local struct to simplify communications better force[...] */
    float x, y, z;
    float vx, vy, vz;
    float *fx, *fy, *fz;
    uint id;
};

__device__ void force0(const Rnd rnd, float2 *pp,
                       int org0,  int org1,  int org2,
                       uint cnt0, uint cnt1, uint cnt2,
                       Part p) {
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
    for (uint i = threadIdx.x & 1; i < cnt2; i += 2) {
        int m1 = (int)(i >= cnt0);
        int m2 = (int)(i >= cnt1);
        uint spid = i + (m2 ? org2 : m1 ? org1 : org0);

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

__device__ void force1(const Frag frag, const Rnd rnd, /**/ Part p) {
    uint cnt0, cnt1, cnt2;
    int org0, org1, org2;
    int count1, count2;
    int basecid, xstencilsize, ystencilsize, zstencilsize;
    int xcid, ycid, zcid;
    int xbasecid, ybasecid, zbasecid;
    int dx, dy, dz;
    dx = frag.dx; dy = frag.dy; dz = frag.dz;

    basecid = 0; xstencilsize = 1; ystencilsize = 1; zstencilsize = 1;
    if (dz == 0) {
        zcid = (int)(p.z + ZS / 2);
        zbasecid = max(0, -1 + zcid);
        basecid = zbasecid;
        zstencilsize = min(frag.zcells, zcid + 2) - zbasecid;
    }

    basecid *= frag.ycells;

    if (dy == 0) {
        ycid = (int)(p.y + YS / 2);
        ybasecid = max(0, -1 + ycid);
        basecid += ybasecid;
        ystencilsize = min(frag.ycells, ycid + 2) - ybasecid;
    }

    basecid *= frag.xcells;

    if (dx == 0) {
        xcid = (int)(p.x + XS / 2);
        xbasecid = max(0, -1 + xcid);
        basecid += xbasecid;
        xstencilsize = min(frag.xcells, xcid + 2) - xbasecid;
    }

    int rowstencilsize = 1, colstencilsize = 1, ncols = 1;

    if (frag.type == FACE) {
        rowstencilsize = dz ? ystencilsize : zstencilsize;
        colstencilsize = dx ? ystencilsize : xstencilsize;
        ncols = dx ? frag.ycells : frag.xcells;
    } else if (frag.type == EDGE)
        colstencilsize = max(xstencilsize, max(ystencilsize, zstencilsize));

    org0 = __ldg(frag.start + basecid);
    cnt0 = __ldg(frag.start + basecid + colstencilsize) - org0;

    org1   = org2 = 0;
    count1 = count2 = 0;
    if (rowstencilsize > 1) {
        org1   = __ldg(frag.start + basecid + ncols);
        count1 = __ldg(frag.start + basecid + ncols + colstencilsize) - org1;
    }

    if (rowstencilsize > 2) {
        org2   = __ldg(frag.start + basecid + 2 * ncols);
        count2 = __ldg(frag.start + basecid + 2 * ncols + colstencilsize) - org2;
    }

    cnt1 = cnt0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    p.x -= dx * XS;
    p.y -= dy * YS;
    p.z -= dz * ZS;

    force0(rnd, frag.pp,
           org0, org1, org2,
           cnt0, cnt1, cnt2,
           p);
}

static __device__ void i2f(const int *ii, float *f, uint i, /**/ float **fx, float **fy, float **fz) {
    f += 3 * ii[i];
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

__device__ void force3(const SFrag sfrag, const Frag frag, const Rnd rnd, uint i, /**/ float *ff) {
    Part p;
    p2rv(sfrag.pp,     i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    i2f (sfrag.ii, ff, i, /**/ &p.fx, &p.fy, &p.fz);
    p.id = i;
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
    force3(sfrag, frag, rnd, i, /**/ ff);
}
}

namespace k_bipsbatch {
static __constant__ unsigned int start[27];

static __constant__ SFrag        ssfrag[26];
static __constant__ Frag          ffrag[26];
static __constant__ Rnd            rrnd[26];
  
__device__ void force1(const Frag frag, const Rnd rnd,
                       uint dpid,
		       float x, float y, float z,
		       float vx, float vy, float vz,
		       /**/ float *fx, float *fy, float *fz) {
    uint cnt0, scan2, ncandidates;
    int org0, org1, org2;
    int basecid, xstencilsize, ystencilsize, zstencilsize;
    int xcid, ycid, zcid;
    int xbasecid, ybasecid, zbasecid;

    org1 = org2 = 0;
    basecid = 0; xstencilsize = 1; ystencilsize = 1; zstencilsize = 1;
    if (frag.dz == 0) {
        zcid = (int)(z + ZS / 2);
        zbasecid = max(0, -1 + zcid);
        basecid = zbasecid;
        zstencilsize = min(frag.zcells, zcid + 2) - zbasecid;
    }

    basecid *= frag.ycells;

    if (frag.dy == 0) {
        ycid = (int)(y + YS / 2);
        ybasecid = max(0, -1 + ycid);
        basecid += ybasecid;
        ystencilsize = min(frag.ycells, ycid + 2) - ybasecid;
    }

    basecid *= frag.xcells;

    if (frag.dx == 0) {
        xcid = (int)(x + XS / 2);
        xbasecid = max(0, -1 + xcid);
        basecid += xbasecid;
        xstencilsize = min(frag.xcells, xcid + 2) - xbasecid;
    }

    x -= frag.dx * XS;
    y -= frag.dy * YS;
    z -= frag.dz * ZS;

    int rowstencilsize = 1, colstencilsize = 1, ncols = 1;

    if (frag.type == FACE) {
        rowstencilsize = frag.dz ? ystencilsize : zstencilsize;
        colstencilsize = frag.dx ? ystencilsize : xstencilsize;
        ncols = frag.dx ? frag.ycells : frag.xcells;
    } else if (frag.type == EDGE)
        colstencilsize = max(xstencilsize, max(ystencilsize, zstencilsize));

    org0 = __ldg(frag.cellstarts + basecid);
    int count0 = __ldg(frag.cellstarts + basecid + colstencilsize) - org0;

    int count1 = 0, count2 = 0;

    if (rowstencilsize > 1) {
        org1 = __ldg(frag.cellstarts + basecid + ncols);
        count1 = __ldg(frag.cellstarts + basecid + ncols + colstencilsize) -
            org1;
    }

    if (rowstencilsize > 2) {
        org2 = __ldg(frag.cellstarts + basecid + 2 * ncols);
        count2 = __ldg(frag.cellstarts + basecid + 2 * ncols + colstencilsize) -
            org2;
    }

    cnt0 = count0;
    scan2 = cnt0 + count1;
    ncandidates = scan2 + count2;

    org1 -= cnt0;
    org2 -= scan2;

    float2 *pp = frag.pp;
    int mask = rnd.mask;
    float seed = rnd.seed;

    float xforce = 0, yforce = 0, zforce = 0;
    for (uint i = threadIdx.x & 1; i < ncandidates; i += 2) {
        int m1 = (int)(i >= cnt0);
        int m2 = (int)(i >= scan2);
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

__device__ void force2(const SFrag sfrag, const Frag frag, const Rnd rnd,
                       uint i,
		       float x, float y, float z,
		       float vx, float vy, float vz,
		       /**/ float *ff) {
    float *fx, *fy, *fz;

    int k;
    k = 3 * sfrag.ii[i];
    fx = &ff[k++];
    fy = &ff[k++];
    fz = &ff[k++];

    force1(frag, rnd, i, x, y, z, vx, vy, vz, /**/ fx, fy, fz);
}


__device__ void force3(const SFrag sfrag, const Frag frag, const Rnd rnd, uint i, /**/ float *ff) {
    float x, y, z, vx, vy, vz;

    int k;
    k  = 6*i;
    x  = sfrag.pp[k++];
    y  = sfrag.pp[k++];
    z  = sfrag.pp[k++];

    vx = sfrag.pp[k++];
    vy = sfrag.pp[k++];
    vz = sfrag.pp[k++];

    force2(sfrag, frag, rnd, i, x, y, z, vx, vy, vz, /**/ ff);
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

namespace k_bipsbatch {
static __constant__ unsigned int start[27];

static __constant__ SFrag        ssfrag[26];
static __constant__ Frag          ffrag[26];
static __constant__ Rnd            rrnd[26];

static __device__ unsigned int get_hid(const unsigned int a[], const unsigned int i) {
    /* where is `i' in sorted a[27]? */
    unsigned int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}
  
__device__ void force0(const Frag frag, const Rnd rnd,
                       uint dpid,
		       float x, float y, float z,
		       float vx, float vy, float vz,
		       /**/ float *fx, float *fy, float *fz) {
    uint scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;
    int basecid, xstencilsize, ystencilsize, zstencilsize;
    int xcid, ycid, zcid;
    int xbasecid, ybasecid, zbasecid;

    deltaspid1 = deltaspid2 = 0;
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

    spidbase = __ldg(frag.cellstarts + basecid);
    int count0 = __ldg(frag.cellstarts + basecid + colstencilsize) - spidbase;

    int count1 = 0, count2 = 0;

    if (rowstencilsize > 1) {
        deltaspid1 = __ldg(frag.cellstarts + basecid + ncols);
        count1 = __ldg(frag.cellstarts + basecid + ncols + colstencilsize) -
            deltaspid1;
    }

    if (rowstencilsize > 2) {
        deltaspid2 = __ldg(frag.cellstarts + basecid + 2 * ncols);
        count2 = __ldg(frag.cellstarts + basecid + 2 * ncols + colstencilsize) -
            deltaspid2;
    }

    scan1 = count0;
    scan2 = scan1 + count1;
    ncandidates = scan2 + count2;

    deltaspid1 -= scan1;
    deltaspid2 -= scan2;

    float2 *xsrc = frag.xsrc;
    int mask = rnd.mask;
    float seed = rnd.seed;

    float xforce = 0, yforce = 0, zforce = 0;
    for (uint i = threadIdx.x & 1; i < ncandidates; i += 2) {
        int m1 = (int)(i >= scan1);
        int m2 = (int)(i >= scan2);
        uint spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

        float2 s0 = __ldg(xsrc + 0 + spid * 3);
        float2 s1 = __ldg(xsrc + 1 + spid * 3);
        float2 s2 = __ldg(xsrc + 2 + spid * 3);

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

__device__ void force1(const Frag frag, const Rnd rnd,
                       uint i,
		       float x, float y, float z,
		       float vx, float vy, float vz,
		       /**/ float *ff) {
    float *fx, *fy, *fz;

    int k;
    k = 3 * frag.ii[i];
    fx = &ff[k++];
    fy = &ff[k++];
    fz = &ff[k++];

    force0(frag, rnd, i, x, y, z, vx, vy, vz, /**/ fx, fy, fz);
}


__device__ void force2(const Frag frag, const Rnd rnd, uint i, /**/ float *ff) {
    float x, y, z, vx, vy, vz;

    int k;
    k  = 6*i;
    x  = frag.xdst[k++];
    y  = frag.xdst[k++];
    z  = frag.xdst[k++];

    vx = frag.xdst[k++];
    vy = frag.xdst[k++];
    vz = frag.xdst[k++];

    force1(frag, rnd, i, x, y, z, vx, vy, vz, /**/ ff);
}

__global__ void force(/**/ float *ff) {
    Frag frag;
    Rnd  rnd;
    int gid;
    uint hid; /* halo id */
    uint i; /* particle id */

    gid = (threadIdx.x + blockDim.x * blockIdx.x) >> 1;
    if (gid >= start[26]) return;
    hid = get_hid(start, gid);
    i = gid - start[hid];
    frag = ffrag[hid];
    if (i >= frag.ndst) return;
    rnd = rrnd[hid];
    force2(frag, rnd, i, /**/ ff);
}
}

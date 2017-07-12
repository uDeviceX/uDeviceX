namespace bipsbatch {
__constant__ unsigned int start[27];
__constant__ BatchInfo batchinfos[26];

static __device__ unsigned int get_hid(const unsigned int a[], const unsigned int i) {  /* where is `i' in sorted a[27]? */
    unsigned int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

  
__device__ void force2(const BatchInfo info, uint dpid, /**/ float *ff) {
    float x, y, z;
    float vx, vy, vz;
    int dstbase;
    uint scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;
    int basecid, xstencilsize, ystencilsize, zstencilsize;


    x = info.xdst[0 + dpid * 6];
    y = info.xdst[1 + dpid * 6];
    z = info.xdst[2 + dpid * 6];

    vx = info.xdst[3 + dpid * 6];
    vy = info.xdst[4 + dpid * 6];
    vz = info.xdst[5 + dpid * 6];

    dstbase = 3 * info.scattered_entries[dpid];

    deltaspid1 = deltaspid2 = 0;
    {
        basecid = 0; xstencilsize = 1; ystencilsize = 1; zstencilsize = 1;
        {
            if (info.dz == 0) {
                int zcid = (int)(z + ZS / 2);
                int zbasecid = max(0, -1 + zcid);
                basecid = zbasecid;
                zstencilsize = min(info.zcells, zcid + 2) - zbasecid;
            }

            basecid *= info.ycells;

            if (info.dy == 0) {
                int ycid = (int)(y + YS / 2);
                int ybasecid = max(0, -1 + ycid);
                basecid += ybasecid;
                ystencilsize = min(info.ycells, ycid + 2) - ybasecid;
            }

            basecid *= info.xcells;

            if (info.dx == 0) {
                int xcid = (int)(x + XS / 2);
                int xbasecid = max(0, -1 + xcid);
                basecid += xbasecid;
                xstencilsize = min(info.xcells, xcid + 2) - xbasecid;
            }

            x -= info.dx * XS;
            y -= info.dy * YS;
            z -= info.dz * ZS;
        }

        int rowstencilsize = 1, colstencilsize = 1, ncols = 1;

        if (info.halotype == FACE) {
            rowstencilsize = info.dz ? ystencilsize : zstencilsize;
            colstencilsize = info.dx ? ystencilsize : xstencilsize;
            ncols = info.dx ? info.ycells : info.xcells;
        } else if (info.halotype == EDGE)
        colstencilsize = max(xstencilsize, max(ystencilsize, zstencilsize));

        spidbase = __ldg(info.cellstarts + basecid);
        int count0 = __ldg(info.cellstarts + basecid + colstencilsize) - spidbase;

        int count1 = 0, count2 = 0;

        if (rowstencilsize > 1) {
            deltaspid1 = __ldg(info.cellstarts + basecid + ncols);
            count1 = __ldg(info.cellstarts + basecid + ncols + colstencilsize) -
                deltaspid1;
        }

        if (rowstencilsize > 2) {
            deltaspid2 = __ldg(info.cellstarts + basecid + 2 * ncols);
            count2 = __ldg(info.cellstarts + basecid + 2 * ncols + colstencilsize) -
                deltaspid2;
        }

        scan1 = count0;
        scan2 = scan1 + count1;
        ncandidates = scan2 + count2;

        deltaspid1 -= scan1;
        deltaspid2 -= scan2;
    }

    float2 *xsrc = info.xsrc;
    int mask = info.mask;
    float seed = info.seed;

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
        float3 pos1 = make_float3(x, y, z), pos2 = make_float3(s0.x, s0.y, s1.x);
        float3 vel1 = make_float3(vx, vy, vz), vel2 = make_float3(s1.y, s2.x, s2.y);
        float3 strength = force(SOLVENT_TYPE, SOLVENT_TYPE, pos1, pos2, vel1,
                                                   vel2, myrandnr);

        xforce += strength.x;
        yforce += strength.y;
        zforce += strength.z;
    }

    atomicAdd(ff + dstbase + 0, xforce);
    atomicAdd(ff + dstbase + 1, yforce);
    atomicAdd(ff + dstbase + 2, zforce);
}

  __global__ void force(float *ff) {
    BatchInfo info;
    int gid;
    uint code, dpid;

    gid = (threadIdx.x + blockDim.x * blockIdx.x) >> 1;
    if (gid >= start[26]) return;
    code = get_hid(start, gid);
    dpid = gid - start[code];
    info = batchinfos[code];
    if (dpid >= info.ndst) return;

    force2(info, dpid, /**/ ff);
  }
}

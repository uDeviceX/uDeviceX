namespace dpdr {
namespace sub {
namespace dev {

/* returns halo box; 0 is a corner of subdomain */
static __device__ void get_box(int i, /**/ int org[3], int ext[3]) {
    /* i, org, ext : fragment id, origin, extend */
    int L[3] = {XS, YS, ZS};
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
    int c;
    for (c = 0; c < 3; ++c) {
        org[c] = (d[c] == 1) ? L[c] - 1 : 0;
        ext[c] = (d[c] == 0) ? L[c]     : 1;
    }
}

/* halo to bulk cell id */
static __device__ int h2cid(int hci, const int org[3], const int ext[3]) {
    enum {X, Y, Z};
    int c;
    int srccellpos[3];
    int dstcellpos[3] = {hci % ext[X], (hci / ext[X]) % ext[Y], hci / (ext[X] * ext[Y])};
    for (c = 0; c < 3; ++c) srccellpos[c] = org[c] + dstcellpos[c];
    return srccellpos[X] + XS * (srccellpos[Y] + YS * srccellpos[Z]);
}

__global__ void count(const int27 cellpackstarts, const int *start, const int *count, /**/
                      intp26 fragss, intp26 fragcc) {
    enum {X, Y, Z};
    int gid;
    int fid; /* fragment id */
    int nhc; /* number of halo cells */
    int cid, hci; /* bulk and halo cell ids */
    int org[3], ext[3]; /* fragment origin and extend */
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= cellpackstarts.d[26]) return;

    fid = k_common::fid(cellpackstarts.d, gid);
    hci = gid - cellpackstarts.d[fid];

    get_box(fid, /**/ org, ext);
    nhc = ext[X] * ext[Y] * ext[Z];

    if (hci < nhc) {
        cid = h2cid(hci, org, ext);
        fragss.d[fid][hci] = start[cid];
        fragcc.d[fid][hci] = count[cid];
    } else if (hci == nhc) {
        fragss.d[fid][hci] = 0;
        fragcc.d[fid][hci] = 0;
    }
}

__global__ void copycells(const int27 cellpackstarts, const intp26 srccells, /**/ intp26 dstcells) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid >= cellpackstarts.d[26]) return;

    int idpack = k_common::fid(cellpackstarts.d, gid);
    int offset = gid - cellpackstarts.d[idpack];

    dstcells.d[idpack][offset] = srccells.d[idpack][offset];
}

template <int NWARPS>
__global__ void scan(const int26 fragn, const intp26 fragcc, /**/ intp26 fragcum) {
    __shared__ int shdata[32];

    int fid = blockIdx.x;
    int *count = fragcc.d[fid];
    int *start = fragcum.d[fid];
    int n = fragn.d[fid];

    int tid = threadIdx.x;
    int laneid = threadIdx.x % warpSize;
    int warpid = threadIdx.x / warpSize;

    int lastval = 0;
    for (int sourcebase = 0; sourcebase < n; sourcebase += 32 * NWARPS) {
        int sourceid = sourcebase + tid;
        int mycount = 0, myscan = 0;
        if (sourceid < n) myscan = mycount = count[sourceid];
        if (tid == 0) myscan += lastval;

        for (int L = 1; L < 32; L <<= 1) {
            int val = __shfl_up(myscan, L);
            if (laneid >= L) myscan += val;
        }

        if (laneid == 31) shdata[warpid] = myscan;
        __syncthreads();
        if (warpid == 0) {
            int gs = 0;
            if (laneid < NWARPS) gs = shdata[tid];
            for (int L = 1; L < 32; L <<= 1) {
                int val = __shfl_up(gs, L);
                if (laneid >= L) gs += val;
            }

            shdata[tid] = gs;
            lastval = __shfl(gs, 31);
        }
        __syncthreads();
        if (warpid) myscan += shdata[warpid - 1];
        __syncthreads();
        if (sourceid < n) start[sourceid] = myscan - mycount;
    }
}
  
__global__ void fill_all(const int27 cellpackstarts, const Particle *pp, const intp26 fragss, 
                         const intp26 fragcc, const intp26 fragcum, const int26 fragcapacity,
                         /**/ intp26 fragindices, Particlep26 fragpp, int *required_bag_size) {
    int gid, fid, hci, tid, src, dst, nsrc, nfloat2s;
    int i, lpid, dpid, spid, c;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* gid: work group id                */
    /* tid: worker id within the group   */
    
    gid = (threadIdx.x >> 4) + 2 * blockIdx.x;
    if (gid >= cellpackstarts.d[26]) return;

    fid = k_common::fid(cellpackstarts.d, gid);
    hci = gid - cellpackstarts.d[fid];
    
    tid = threadIdx.x & 0xf;
    src = fragss.d[fid][hci];
    dst = fragcum.d[fid][hci];
    nsrc = min(fragcc.d[fid][hci], fragcapacity.d[fid] - dst);

    const float2 *srcpp = (const float2*) pp;
    float2 *dstpp = (float2*) fragpp.d[fid];
    
    nfloat2s = nsrc * 3;
    for (i = tid; i < nfloat2s; i += warpSize/2) {
        lpid = i / 3;
        c    = i % 3;
        dpid = dst + lpid;
        spid = src + lpid;
        dstpp[3 * dpid + c] = srcpp[3 * spid + c];
    }
    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fragindices.d[fid][dpid] = spid;
    }
    if (gid + 1 == cellpackstarts.d[fid + 1]) required_bag_size[fid] = dst;
}

__global__ void fill_all_ii(const int27 cellpackstarts, const int *ii,
                            const intp26 fragss, const intp26 fragcc, const intp26 fragcum,
                            const int26 fragcapacity, /**/ intp26 fragii) {
    int gid, fid, hci, tid, src, dst, nsrc;
    int lpid, dpid, spid;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* gid: work group id                */
    /* tid: worker id within the group   */
    
    gid = (threadIdx.x >> 4) + 2 * blockIdx.x;
    if (gid >= cellpackstarts.d[26]) return;

    fid = k_common::fid(cellpackstarts.d, gid);
    hci = gid - cellpackstarts.d[fid];
    
    tid = threadIdx.x & 0xf;
    src = fragss.d[fid][hci];
    dst = fragcum.d[fid][hci];
    nsrc = min(fragcc.d[fid][hci], fragcapacity.d[fid] - dst);
    
    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fragii.d[fid][dpid] = ii[spid];
    }
}

} // dev
} // sub
} // dpdr

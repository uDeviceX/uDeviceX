namespace dev {

__global__ void build_map(int3 L, int n, const float3 *minext, const float3 *maxext, /**/ Map map) {
    int i, fid, fids[MAX_DSTS], ndsts, j;
    float3 lo, hi;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    lo = minext[i];
    hi = maxext[i];
    
    fid = map_code_box(L, lo, hi);
    ndsts = map_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        add_to_map(0, i, fids[j], /**/ map);
}

static __device__ void pack_p(int nv, const Particle *pp, int vid, int frag_mid, int *indices, /**/ Particle *buf) {
    int dst, src, mid;
    mid = __ldg(indices + frag_mid);
    dst = nv * frag_mid + vid;
    src = nv * mid      + vid;
    buf[dst] = pp[src];
}

__global__ void pack_mesh(int nv, const Particle *pp, Map map, /**/ Pap26 buf) {
    int gid, hi, step, fid, mid, vid, frag_mid;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = map.starts[26] * nv;
    step = gridDim.x * blockDim.x;
    
    for (  ; gid < hi; gid += step) {
        mid = gid / nv; /* mesh id   */
        vid = gid % nv; /* vertex id */
        fid = k_common::fid(map.starts, mid);

        /* mesh index in the fragment coordinates */ 
        frag_mid = mid - map.starts[fid];
        
        pack_p(nv, pp, vid, frag_mid, map.ids[fid], /**/ buf.d[fid]);
    }
}

/* compress kernels */

enum {SKIP=-1};

// TODO this is copy paste from meshbb, might be removed from there
static __device__ bool nz(float a) {return fabs(a) > 1e-6f;}

static __device__ bool nonzero(const Momentum *m) {
    enum {X, Y, Z};
    return nz(m->P[X]) && nz(m->P[Y]) && nz(m->P[Z]) &&
        nz(m->L[X]) && nz(m->L[Y]) && nz(m->L[Z]);
}

__global__ void subindex_compress(int nt, int nm, const Momentum *mm, /**/ int *counts, int *subids) {
    int i, mid, subid;
    Momentum m;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nt * nm) return;
    
    mid = i / nt;

    m = mm[i];

    if (nonzero(&m))
        subid = atomicAdd(counts + mid, 1);
    else
        subid = SKIP;
    subids[i] = subid;
}

/* inclusive scan */
static __device__ int warpScan(int val) {
    int tid;
    tid = threadIdx.x % warpSize;
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    return val;
}

template<int NWARP>
__global__ void block_scan(int n, const int *cc, /**/ int *ss) {
    assert(n < blockDim.x);
    assert(gridDim.x == 1);
    //static_assert(NWARP <= 32);
    int i, tid, wid, lid, val, cnt, ws;
    tid = threadIdx.x;
    wid = tid / warpSize;
    lid = tid % warpSize;
    
    i = tid + blockIdx.x * blockDim.x;

    __shared__ int warp_cnt[NWARP];

    cnt = val = 0;

    if (i < n) cnt = cc[i];

    val = warpScan(cnt);

    if (lid == warpSize - 1) warp_cnt[wid] = val;

    __syncthreads();

    if (wid == 0) {
        int v, c;
        v = c = 0;
        if (lid < NWARP) c = warp_cnt[lid];
        v = warpScan(c);
        if (lid < NWARP) warp_cnt[lid] = v - c;
    }

    __syncthreads();
    
    ws = warp_cnt[wid];

    val += ws;
    
    if (i < n) ss[i] = val - cnt;
}

__global__ void compress(int nt, int nm, const Momentum *mm, const int *starts, const int *subids, /**/ int *ids, Momentum *mmc) {
    int i, mid, subid, entry;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nt * nm) return;
    
    mid = i / nt;

    subid = subids[i];

    if (subid == SKIP) return;
    
    entry = subid + starts[mid];
    mmc[entry] = mm[i];
    ids[i] = i;
}

} // dev

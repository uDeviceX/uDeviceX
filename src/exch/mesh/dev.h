namespace emesh_dev {

/* L0: subdomain cropped by one cutoff radius */
__global__ void build_map(int3 L0, int n, const float3 *minext, const float3 *maxext, /**/ EMap map) {
    int i, fid, fids[MAX_DSTS], ndsts, j;
    float3 lo, hi;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    lo = minext[i];
    hi = maxext[i];
    
    fid = emap_code_box(L0, lo, hi);
    ndsts = emap_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        emap_add(0, i, fids[j], /**/ map);
}

static __device__ void pack_p(int nv, const Particle *pp, int vid, int frag_mid, int *indices, /**/ Particle *buf) {
    int dst, src, mid;
    mid = __ldg(indices + frag_mid);
    dst = nv * frag_mid + vid;
    src = nv * mid      + vid;
    buf[dst] = pp[src];
}

__global__ void pack_mesh(int nv, const Particle *pp, EMap map, /**/ Pap26 buf) {
    int gid, hi, step, fid, mid, vid, frag_mid;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = map.starts[26] * nv;
    step = gridDim.x * blockDim.x;
    
    for (  ; gid < hi; gid += step) {
        mid = gid / nv; /* mesh id   */
        vid = gid % nv; /* vertex id */
        fid = frag_dev::frag_get_fid(map.starts, mid);

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

/* exclusive scan on one block */
template<int NWARP>
__global__ void block_scan(int n, const int *cc, /**/ int *ss) {
    assert(n < blockDim.x);
    assert(gridDim.x == 1);

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
    
    if (i <= n) ss[i] = val - cnt;
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

__global__ void collect_counts(int26 nm, MMap26 mm, /**/ int *counts) {
    int i, c, end;
    i = threadIdx.x;
    if (i > 26) return;
    end = nm.d[i];
    c = mm.d[i].ss[end];
    counts[i] = c;
}

static __device__ void addMom(const Momentum src, /**/ Momentum *dst) {
    enum {X, Y, Z};
    atomicAdd(dst->P + X, src.P[X]);
    atomicAdd(dst->P + Y, src.P[Y]);
    atomicAdd(dst->P + Z, src.P[Z]);

    atomicAdd(dst->L + X, src.L[X]);
    atomicAdd(dst->L + Y, src.L[Y]);
    atomicAdd(dst->L + Z, src.L[Z]);
}

static __device__ void unpack_mom_frag(int nt, int i, const int *hii, const Momentum *hmm, const int *mapii, /**/ Momentum *mm) {
    int hid, hmid, tid, lmid, lid;
    
    hid = hii[i];        /* global index in frag                */
    hmid = hid / nt;     /* mesh index in frag                  */
    tid  = hid % nt;     /* triangle id (same in frag or local) */

    lmid = mapii[hmid];  /* mesh index in bulk                  */

    lid = lmid * nt + tid;

    addMom(hmm[i], /**/ mm + lid);
}

__global__ void unpack_mom(int nt, int27 fragstarts, intp26 hii, Mop26 hmm, EMap map, /**/ Momentum *mm) {
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    fid = frag_dev::frag_get_fid(fragstarts.d, i);

    if (i >= fragstarts.d[26]) return;

    i -= fragstarts.d[fid];
    
    unpack_mom_frag(nt, i, hii.d[fid], hmm.d[fid], map.ids[fid], /**/ mm);
}

} // dev

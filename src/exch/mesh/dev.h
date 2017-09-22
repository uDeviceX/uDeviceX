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

// TODO this is copy/paste from distr/common
static __device__ void fid2shift(int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = XS * frag_i2d(id, X);
    s[Y] = YS * frag_i2d(id, Y);
    s[Z] = ZS * frag_i2d(id, Z);
}

static  __device__ void shift_1p(const int s[3], /**/ Particle *p) {
    enum {X, Y, Z};
    p->r[X] += s[X];
    p->r[Y] += s[Y];
    p->r[Z] += s[Z];
}

__global__ void shift_one_frag(int n, const int fid, /**/ Particle *pp) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    
    fid2shift(fid, /**/ s);
    shift_1p(s, /**/ pp + i);
}

} // dev

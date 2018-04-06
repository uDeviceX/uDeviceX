static __device__ void pack_p(const Particle *pp, int offset, int *indices, int cap, int i, /**/ Particle *buf) {
    int dst, src;
    dst = offset + i;
    if (dst >= cap) return;
    src = __ldg(indices + dst);
    buf[dst] = pp[src];
}

__global__ void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf) {
    int gid, fid, frag_i, hi, step;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = ph.starts[26];
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        fid = frag_dev::frag_get_fid(ph.starts, gid);

        /* index in the fragment coordinates */ 
        frag_i = gid - ph.starts[fid];
        
        pack_p(pp, ph.offsets[fid], ph.indices[fid], ph.cap[fid], frag_i, /**/ buf.d[fid]);
    }
}

// TODO this is copy/paste from distr/common
static __device__ void fid2shift(int3 L, int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = L.x * frag_dev::i2dx(id);
    s[Y] = L.y * frag_dev::i2dy(id);
    s[Z] = L.z * frag_dev::i2dz(id);
}

static  __device__ void shift_1p(const int s[3], /**/ Particle *p) {
    enum {X, Y, Z};
    p->r[X] += s[X];
    p->r[Y] += s[Y];
    p->r[Z] += s[Z];
}

__global__ void ecommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    
    fid2shift(L, fid, /**/ s);
    shift_1p(s, /**/ pp + i);
}

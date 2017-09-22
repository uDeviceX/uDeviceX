namespace dev {

static __device__ void pack_p(const Particle *pp, int offset, int *indices, int i, /**/ Particle *buf) {
    int dst, src;
    dst = offset + i;
    src = __ldg(indices + dst);
    buf[dst] = pp[src];
}

__global__ void pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf) {
    int gid, fid, frag_i, hi, step;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = ph.starts[26];
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        fid = k_common::fid(ph.starts, gid);

        /* index in the fragment coordinates */ 
        frag_i = gid - ph.starts[fid];
        
        pack_p(pp, ph.offsets[fid], ph.indices[fid], frag_i, /**/ buf.d[fid]);
    }
}

static __device__ void unpack_f(const Force *hff, int offset, int *indices, int i, /**/ Force *ff) {
    enum {X, Y, Z};
    int dst, src;

    src = offset + i;
    dst = __ldg(indices + src);

    const Force f = hff[src];
    
    atomicAdd(ff[dst].f + X, f.f[X]);
    atomicAdd(ff[dst].f + Y, f.f[Y]);
    atomicAdd(ff[dst].f + Z, f.f[Z]);
}

__global__ void unpack_ff(Fop26 hff, PackHelper ph, /**/ Force *ff) {
    int gid, fid, frag_i, hi, step;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = ph.starts[26];
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        fid = k_common::fid(ph.starts, gid);

        /* index in the fragment coordinates */ 
        frag_i = gid - ph.starts[fid];
        
        unpack_f(hff.d[fid], ph.offsets[fid], ph.indices[fid], frag_i, /**/ ff);
    }
}

} // dev

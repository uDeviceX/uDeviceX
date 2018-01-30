namespace dev {

__global__ void build_map(int3 L, int soluteid, int n, const Particle *pp, /**/ EMap map) {
    int pid, fid, fids[MAX_DSTS], ndsts, j;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    const Particle p = pp[pid];

    fid = map_code(L, p.r);
    ndsts = map_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        add_to_map(soluteid, pid, fids[j], /**/ map);
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
        fid = fragdev::frag_get_fid(ph.starts, gid);

        /* index in the fragment coordinates */ 
        frag_i = gid - ph.starts[fid];
        
        unpack_f(hff.d[fid], ph.offsets[fid], ph.indices[fid], frag_i, /**/ ff);
    }
}

} // dev

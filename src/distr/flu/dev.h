namespace dev {

/* TODO put in frag.h */
__device__ int d2fid(const int d[3]) {
    return ((d[0] + 2) % 3)
        + 3 * ((d[1] + 2) % 3)
        + 9 * ((d[2] + 2) % 3);
}


__global__ void scan_map(/**/ Map m) {
    int tid, val;
    tid = threadIdx.x;
    val = 0;    

    if (tid < 26) val = m.counts[tid];    
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    if (tid < 27) m.starts[tid] = val;
}

__device__ int get_fid(const float r[3]) {
    enum {X, Y, Z};
    int d[3];
    d[X] = -1 + (r[X] >= -XS/2) + (r[X] >= XS/2);
    d[Y] = -1 + (r[Y] >= -YS/2) + (r[Y] >= YS/2);
    d[Z] = -1 + (r[Z] >= -ZS/2) + (r[Z] >= ZS/2);
    return d2fid(d);
}

__device__ void add_to_map(int pid, int fid, Map m) {
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}

__global__ void build_map(const Particle *pp, const int n, /**/ Map m) {
    int pid, fid;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    const Particle p = pp[pid];

    fid = get_fid(p.r);

    if (fid != frag_bulk)
        add_to_map(pid, fid, /**/ m);
}


template <typename T, int STRIDE>
__global__ void pack(const T *data, Map m, /**/ T *buf[]) {
    int gid, slot;
    int fid; /* [f]ragment [id] */
    int offset, pid, c, d, s;
    
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    slot = gid / STRIDE;
    fid = k_common::fid(m.starts, slot);
    if (slot >= m.starts[27]) return;
    c = gid % STRIDE;

    offset = slot - m.starts[fid];
    pid = __ldg(m.ids[fid] + offset);

    d = c + STRIDE * offset;
    s = c + STRIDE * pid;
    
    buf[fid][d] = data[s];
}

} // dev

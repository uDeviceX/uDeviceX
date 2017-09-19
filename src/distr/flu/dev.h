namespace dev {

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
__global__ void pack(const T *data, Map m, /**/ Sarray<T*, 26> buf) {
    int gid, slot;
    int fid; /* [f]ragment [id] */
    int offset, pid, c, d, s;
    
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    slot = gid / STRIDE;
    fid = k_common::fid(m.starts, slot);
    if (slot >= m.starts[26]) return;
    c = gid % STRIDE;

    offset = slot - m.starts[fid];
    pid = __ldg(m.ids[fid] + offset);

    d = c + STRIDE * offset;
    s = c + STRIDE * pid;
    
    buf.d[fid][d] = data[s];
}

} // dev

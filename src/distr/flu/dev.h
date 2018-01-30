namespace dev {

__global__ void build_map(int3 L, const PartList lp, const int n, /**/ DMap m) {
    int pid, fid;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    if (is_dead(pid, lp)) return;

    const Particle p = lp.pp[pid];

    fid = dmap_get_fid(L, p.r);

    if (fid != frag_bulk)
        dmap_add(pid, fid, /**/ m);
}

template <typename T, int STRIDE>
__global__ void pack(const T *data, DMap m, /**/ Sarray<T*, 26> buf) {
    int gid, slot;
    int fid; /* [f]ragment [id] */
    int offset, pid, c, d, s;
    
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    slot = gid / STRIDE;
    fid = fragdev::frag_get_fid(m.starts, slot);
    if (slot >= m.starts[26]) return;
    c = gid % STRIDE;

    offset = slot - m.starts[fid];
    pid = __ldg(m.ids[fid] + offset);

    d = c + STRIDE * offset;
    s = c + STRIDE * pid;
    
    buf.d[fid][d] = data[s];
}

} // dev

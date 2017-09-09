namespace dev {
__device__ int get_fid(const float r[3]) {
    enum {X, Y, Z};
    int dx, dy, dz;
    dx = -1 + (r[X] >= -XS/2) + (r[X] >= XS/2);
    dy = -1 + (r[Y] >= -YS/2) + (r[Y] >= YS/2);
    dz = -1 + (r[Z] >= -ZS/2) + (r[Z] >= ZS/2);
    return frag_to_id(dx, dy, dz);
}

__device__ void add_to_map(int pid, int fid, Map *m) {
    int entry;
    entry = atomicAdd(m->counts + fid, 1);
    m->ids[fid][entry] = pid;
}

__global__ void build_map(const Particle *pp, const int n, /**/ Map *m) {
    int pid, fid;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    const Particle p = pp[pid];

    fid = get_fid(p.r);

    if (fid != frag_bulk)
        add_to_map(pid, fid, /**/ m);
}

} // dev

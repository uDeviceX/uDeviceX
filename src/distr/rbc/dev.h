/* exclusive scan */
__global__ void scan_map(/**/ Map m) {
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0;    

    if (tid < NBAGS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    if (tid <= NBAGS) m.starts[tid] = val - cnt;
}

static __device__ int get_fid(const float r[3]) {
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -XS/2) + (r[X] >= XS/2);
    y = -1 + (r[Y] >= -YS/2) + (r[Y] >= YS/2);
    z = -1 + (r[Z] >= -ZS/2) + (r[Z] >= ZS/2);
    return frag_d2i(x, y, z);
}

static __device__ void add_to_map(int pid, int fid, Map m) {
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}

__global__ void build_map(const float *rr, const int n, /**/ Map m) {
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    const float *r = rr + 3 * i;

    fid = get_fid(r);

    if (fid != frag_bulk)
        add_to_map(pid, fid, /**/ m);
}

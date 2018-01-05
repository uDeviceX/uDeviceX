namespace dev {

/* exclusive scan */
template <int NCOUNTS> 
__global__ void scan_map(/**/ DMap m) {
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0, cnt = 0;    

    if (tid < NCOUNTS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    if (tid <= NCOUNTS) m.starts[tid] = val - cnt;
}

__device__ int get_fid(const float r[3]) {
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -XS/2) + (r[X] >= XS/2);
    y = -1 + (r[Y] >= -YS/2) + (r[Y] >= YS/2);
    z = -1 + (r[Z] >= -ZS/2) + (r[Z] >= ZS/2);
    return frag_d2i(x, y, z);
}

__device__ void add_to_map(int pid, int fid, DMap m) {
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}

} // dev

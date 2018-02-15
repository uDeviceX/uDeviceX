// tag::int[]
/* exclusive scan */
template <int NCOUNTS> 
static __global__ void dmap_scan(/**/ DMap m)              // <1>
// end::int[]
{
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0, cnt = 0;    

    if (tid < NCOUNTS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    if (tid <= NCOUNTS) m.starts[tid] = val - cnt;
}

// tag::int[]
static __device__ int dmap_get_fid(int3 L, const float r[3])       // <2>
// end::int[]
{
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -L.x/2) + (r[X] >= L.x/2);
    y = -1 + (r[Y] >= -L.y/2) + (r[Y] >= L.y/2);
    z = -1 + (r[Z] >= -L.z/2) + (r[Z] >= L.z/2);
    return frag_dev::d2i(x, y, z);
}

// tag::int[]
static __device__ void dmap_add(int pid, int fid, DMap m)  // <3>
// end::int[]
{
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}

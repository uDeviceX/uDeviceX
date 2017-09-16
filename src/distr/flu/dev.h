namespace dev {

/* exclusive scan */
__global__ void scan_map(/**/ Map m) {
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0;    

    if (tid < NFRAGS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    if (tid <= NFRAGS) m.starts[tid] = val - cnt;
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

static __device__ void fid2shift(int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = XS * frag_i2d(id, X);
    s[Y] = YS * frag_i2d(id, Y);
    s[Z] = ZS * frag_i2d(id, Z);
}

static __device__ void check(const float r[3]) {
    enum {X, Y, Z};
    if (r[X] < -XS/2 || r[X] >= XS/2) printf("x out of range: %f\n", r[X]);
    if (r[Y] < -YS/2 || r[Y] >= YS/2) printf("y out of range: %f\n", r[Y]);
    if (r[Z] < -ZS/2 || r[Z] >= ZS/2) printf("z out of range: %f\n", r[Z]);
}

static  __device__ void shift_1p(const int s[3], /**/ Particle *p) {
    enum {X, Y, Z};
    p->r[X] += s[X];
    p->r[Y] += s[Y];
    p->r[Z] += s[Z];
    check(p->r);
}

__global__ void shift(const int27 starts, /**/ Particle *pp) {
    int pid, fid, s[3];

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= starts.d[26]) return;
    fid = k_common::fid(starts.d, pid);
    
    fid2shift(fid, s);
    shift_1p(s, /**/ pp + pid);
}

} // dev

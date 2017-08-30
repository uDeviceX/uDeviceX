namespace odstr { namespace sub { namespace dev {

/* pack an index and [l]ocal/[r]emove flag into one uint */
__device__ void lr_set(int i, bool rem, /**/ uint *u) {
}
__device__ int  lr_get(uint u, /**/ bool *rem, int *i) {
}

__device__ void warpco(/**/ int *ws, int *dw) { /* warp [co]ordinates */
    /* ws: start, dw: shift (lane) */
    int warp;
    warp = threadIdx.x / warpSize;
    *dw   = threadIdx.x % warpSize;
    *ws   = warpSize * warp + blockDim.x * blockIdx.x;
}

struct Part {
    float2 d0, d1, d2;
};

struct Lo { /* particle [lo]cation in memory
               d: shift in wrap, used for collective access  */
    float2 *p;
    int d;
};

__device__ void pp2Lo(float2 *pp, int n, int ws, /**/ Lo *l) {
    int dwe; /* warp or buffer end relative to wrap start (`ws') */
    const int N_FLOAT2_PER_PARTICLE = 3;
    dwe  = min(warpSize, n - ws);
    l->p = pp + N_FLOAT2_PER_PARTICLE * ws;
    l->d = dwe;
}

__device__ int endLo(Lo *l, int d) { /* is `d' behind the end? */
    /* `d' relative to wrap start */
    return d >= l->d;
}

__device__ void readPart(Lo l, /**/ Part *p) {
    k_read::AOS6f(l.p, l.d, /**/ p->d0, p->d1, p->d2);
}

__device__ void writePart(Part *p, /**/ Lo l) {
    k_write::AOS6f(/**/ l.p, l.d, /*i*/ p->d0, p->d1, p->d2);
}

__global__ void scan(const int n, const int size[], /**/ int strt[], int size_pin[]) {
    int tid = threadIdx.x;
    int val = 0, cnt = 0;

    if (tid < 27) {
        val = cnt = size[threadIdx.x];
        if (tid > 0) size_pin[tid] = cnt;
    }

    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    if (tid < 28) strt[tid] = val - cnt;
    if (tid == 26) {
        strt[tid + 1] = val;
        int nbulk = n - val;
        size_pin[0] = nbulk;
    }
}

template <typename T, int STRIDE>
__global__ void pack(const T *data, int *const iidx[], const int start[], /**/ T *buf[]) {
    int gid, slot;
    int fid; /* [f]ragment [id] */
    int offset, pid, c, d, s;
    
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    slot = gid / STRIDE;
    fid = k_common::fid(start, slot);
    if (slot >= start[27]) return;
    c = gid % STRIDE;

    offset = slot - start[fid];
    pid = __ldg(iidx[fid] + offset);

    d = c + STRIDE * offset;
    s = c + STRIDE * pid;
    
    buf[fid][d] = data[s];
}

template <typename T, int STRIDE>
__global__ void unpack(T *const buf[], const int start[], /**/ T *data) {
    int gid, slot;
    int fid; /* [f]ragment [id] */
    int offset, c, d;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    slot = gid / STRIDE;
    fid = k_common::fid(start, slot);
    if (slot >= start[27]) return;
    c = gid % STRIDE;

    offset = slot - start[fid];
    d = c + STRIDE * offset;

    data[gid] = buf[fid][d];
}

__global__ void scatter(const bool remote, const uchar4 *subi, const int n, const int *start,
                        /**/ uint *iidx) {
    uint pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    uchar4 entry = subi[pid];
    int subindex = entry.w;

    if (subindex != 255) {
        int cid = entry.x + XS * (entry.y + YS * entry.z);
        int base = __ldg(start + cid);

        pid |= remote << 31;
        iidx[base + subindex] = pid;
    }
}

static __device__
void xchg_aos2f(int srclane0, int srclane1, int start, float *s0, float *s1) {
    float t0 = __shfl(*s0, srclane0);
    float t1 = __shfl(*s1, srclane1);

    *s0 = start == 0 ? t0 : t1;
    *s1 = start == 0 ? t1 : t0;
    *s1 = __shfl_xor(*s1, 1);
}

static __device__
void xchg_aos4f(int srclane0, int srclane1, int start, float3 *s0, float3 *s1) {
    xchg_aos2f(srclane0, srclane1, start, &s0->x, &s1->x);
    xchg_aos2f(srclane0, srclane1, start, &s0->y, &s1->y);
    xchg_aos2f(srclane0, srclane1, start, &s0->z, &s1->z);
}

__global__ void gather_id(const int *ii_lo, const int *ii_re, int n, const uint *iidx, /**/ int *ii) {
    int spid, data;
    const int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;

    spid = iidx[pid];
    
    const bool remote = (spid >> 31) & 1;
    spid &= ~(1 << 31);
    if (remote) data = ii_re[spid];
    else        data = ii_lo[spid];

    ii[pid] = data;
}

}}} /* namespace */


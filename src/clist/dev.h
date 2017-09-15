namespace dev {

enum {
    VALID   = 0,
    INVALID = 255
};

static __device__ bool valid_cell(int ix, int iy, int iz, int3 ncells) {
    return
        (ix >= 0) && (ix < ncells.x) &&
        (iy >= 0) && (iy < ncells.y) &&
        (iz >= 0) && (iz < ncells.z);
}

static __device__ uchar4 get_entry(const float *r, int3 ncells) {
    enum {X, Y, Z};
    uchar4 e;
    int ix, iy, iz;
    
    ix = (int)floor(r[X] + ncells.x/2);
    iy = (int)floor(r[Y] + ncells.y/2);
    iz = (int)floor(r[Z] + ncells.z/2);

    if (valid_cell(ix, iy, iz, ncells)) e.w =   VALID;
    else                                e.w = INVALID;

    e.x = ix; e.y = iy; e.z = iz;
    return e;
}

static __device__ int get_cid(int3 ncells, uchar4 e) {
    return e.x + ncells.x * (e.y + ncells.y * e.z);
}

__global__ void subindex(int3 ncells, int n, const Particle *pp, /*io*/ int *counts, /**/ uchar4 *ee) {
    int i, cid;
    Particle p;
    uchar4 e;
    
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    p = pp[i];

    e = get_entry(p.r, ncells);
    cid = get_cid(ncells, e);

    if (e.w != INVALID)
        e.w = atomicAdd(counts + cid, 1);

    ee[i] = e;
}

/* used for debugging purpose */
__global__ void ini_ids(int n, /**/ uint *ii) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    ii[i] = 1<<30;
}

/* [l]ocal [r]emote encoder, decoder: pack into one int the type (l/r) and the id */
__device__ void lr_set(int i, bool rem, /**/ uint *u) {
    *u  = (uint) i;
    *u |= rem << 31;
}
__device__ int  lr_get(uint u, /**/ bool *rem) {
    *rem = (u >> 31) & 1;
    u &= ~(1 << 31);
    return (int) u;
}

__global__ void get_ids(bool remote, int3 ncells, int n, const int *starts, const uchar4 *ee, /**/ uint *ii) {
    int i, cid, id, start;
    uint val;
    uchar4 e;
    
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    e = ee[i];
    cid = get_cid(ncells, e);

    if (e.w != INVALID) {
        start = starts[cid];
        id = start + e.w;
        lr_set(i, remote, /**/ &val);
        ii[id] = val;
    }
}

template <typename T>
__device__ void fetch(const T *ddlo, const T *ddre, uint i, /**/ T *d) {
    bool remote; int src;
    src = lr_get(i, /**/ &remote);
    if (remote) *d = ddre[src];
    else        *d = ddlo[src];
}

template <typename T>
__global__ void gather(const T *ddlo, const T *ddre, const uint *ii, int n, /**/ T *dd) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    uint code = ii[i];
    fetch(ddlo, ddre, code, /**/ dd + i);
}

} // dev

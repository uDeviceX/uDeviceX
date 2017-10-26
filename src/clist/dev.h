namespace dev {

enum {
    VALID   = 0,
    INVALID = 255
};

static __device__ bool inside(const float r[3], int3 L) {
    enum {X, Y, Z};
    return
        (r[X] >= -L.x/2) && (r[X] < L.x/2) &&
        (r[Y] >= -L.y/2) && (r[Y] < L.y/2) &&
        (r[Z] >= -L.z/2) && (r[Z] < L.z/2)  ;
}

static __device__ int project_cid(int i, const int Lh) {
    return i < -Lh ? -Lh : i >= Lh ? Lh - 1 : i;
}

static __device__ uchar4 get_entry(const bool project, const float *r, int3 L) {
    enum {X, Y, Z};
    uchar4 e;
    int ix, iy, iz;

    /* must be done in double precision */
    ix = (int) ((double) r[X] + L.x/2);
    iy = (int) ((double) r[Y] + L.y/2);
    iz = (int) ((double) r[Z] + L.z/2);

    if (project) {
        ix = project_cid(ix, L.x/2);
        iy = project_cid(iy, L.y/2);
        iz = project_cid(iz, L.z/2);
    }
    
    if (project || inside(r, L)) e.w =   VALID;
    else                         e.w = INVALID;

    e.x = ix; e.y = iy; e.z = iz;
    return e;
}

static __device__ int get_cid(int3 L, uchar4 e) {
    return e.x + L.x * (e.y + L.y * e.z);
}

/* project:: if particle is outside of domain L, include it in closest cell */
__global__ void subindex(bool project, int3 L, int n, const Particle *pp, /*io*/ int *counts, /**/ uchar4 *ee) {
    int i, cid;
    Particle p;
    uchar4 e;
    
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    p = pp[i];

    e = get_entry(project, p.r, L);
    cid = get_cid(L, e);

    if (e.w != INVALID)
        e.w = atomicAdd(counts + cid, 1);

    ee[i] = e;
}

__global__ void get_ids(int array_id, int3 L, int n, const int *starts, const uchar4 *ee, /**/ uint *ii) {
    int i, cid, id, start;
    uchar4 e;
    
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    e = ee[i];
    cid = get_cid(L, e);

    if (e.w != INVALID) {
        start = starts[cid];
        id = start + e.w;
        ii[id] = encode_id(array_id, i);
    }
}

template <typename T, int N>
__device__ void fetch(const Sarray<const T*, N> src, uint i, /**/ T *d) {
    int src_id, array_id;
    decode_id(i, /**/ &array_id, &src_id);
    *d = src.d[array_id][src_id];
}

template <typename T, int N>
__global__ void gather(const Sarray<const T*, N> src, const uint *ii, int n, /**/ T *dd) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    uint code = ii[i];
    fetch(src, code, /**/ dd + i);
}

} // dev

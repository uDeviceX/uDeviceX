
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
    
    ix = (int)floor(r[X] + 0.5*ncells.x);
    iy = (int)floor(r[Y] + 0.5*ncells.y);
    iz = (int)floor(r[Z] + 0.5*ncells.z);

    if (valid_cell(ix, iy, iz, ncells)) e.w =   VALID;
    else                                e.w = INVALID;

    e.x = ix; e.y = iy; e.z = iz;
    return e;
}

static __device__ int get_cid(int3 ncells, uchar4 e) {
    return e.x + ncells.x * (e.y + ncells.y * e.z);
}

__global__ void subindex(const int n, const Particle *pp, int3 ncells, /*io*/ int *counts, /**/ uchar4 *ee) {
    int i, cid;
    Particle p;
    uchar4 e;
    
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    p = pp[i];

    e = get_entry(p.r, ncells);
    cid = get_cid(ncells, e);

    if (e.w == VALID)
        e.w = atomicAdd(counts + cid, 1);
}

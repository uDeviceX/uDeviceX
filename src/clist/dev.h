static __device__ int encode(int ix, int iy, int iz, int3 ncells) {
    return ix + ncells.x * (iy + iz * ncells.y);
}

static __device__ int get_cid(const float *r, int3 ncells) {
    enum {X, Y, Z};
    int ix = (int)floor(r[X] + 0.5*ncells.x);
    int iy = (int)floor(r[Y] + 0.5*ncells.y);
    int iz = (int)floor(r[Z] + 0.5*ncells.z);

    ix = min(ncells.x - 1, max(0, ix));
    iy = min(ncells.y - 1, max(0, iy));
    iz = min(ncells.z - 1, max(0, iz));
    
    return encode(ix, iy, iz, ncells);
}

__global__ void get_counts(const Particle *pp, const int n, int3 ncells, /**/ int *counts) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    const Particle p = pp[i];
    const int cid = get_cid(p.r, ncells);

    atomicAdd(counts + cid, 1);
}

__global__ void get_ids(const Particle *pp, const int *starts, const int n, int3 ncells, /**/ int *counts, int *ids) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    const Particle p = pp[i];
    int cid = get_cid(p.r, ncells);
    const int start = starts[cid];

    const int id = start + atomicAdd(counts + cid, 1);
    ids[i] = id;
}

__global__ void gather(const Particle *pps, const int *ids, int n, /**/ Particle *ppd) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    const Particle p = pps[i];
    const int     id = ids[i];

    ppd[id] = p;
}

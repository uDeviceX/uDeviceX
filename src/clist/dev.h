enum {X, Y, Z};

static __device__ int encode(int ix, int iy, int iz, int3 ncells) {
  return ix + ncells.x * (iy + iz * ncells.y);
}

static __device__ int get_cid(const float *r, int3 ncells, int3 domainstart) {
    int ix = (int)floor(r[X] - domainstart.x);
    int iy = (int)floor(r[Y] - domainstart.y);
    int iz = (int)floor(r[Z] - domainstart.z);

    ix = min(ncells.x, max(0, ix));
    iy = min(ncells.y, max(0, iy));
    iz = min(ncells.z, max(0, iz));
    
    return encode(ix, iy, iz, ncells);
}

__global__ void get_counts(const Particle *pp, const int n, int3 ncells, int3 domainstart, /**/ int *counts) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    const Particle p = pp[i];
    const int cid = get_cid(p.r, ncells, domainstart);

    atomicAdd(counts + cid, 1);
}

__global__ void get_ids(const Particle *pp, const int *starts, const int n, int3 ncells, int3 domainstart, /**/ int *counts, int *ids) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    const Particle p = pp[i];
    int cid = get_cid(p.r, ncells, domainstart);
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


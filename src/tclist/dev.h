namespace dev
{

template <typename T> __device__ T min3(T a, T b, T c) {return min(a, min(b, c));}
template <typename T> __device__ T max3(T a, T b, T c) {return max(a, max(b, c));}

static  __device__ float3 loadr(const Particle *pp, int i) {
    Particle p = pp[i];
    enum {X, Y, Z};
    return make_float3(p.r[X], p.r[Y], p.r[Z]);
}

static  __device__ int3 loadt(const int4 *tt, int i) {
    int4 t = tt[i];
    return make_int3(t.x, t.y, t.z);
}

static  __device__ void tbbox(const float3 A, const float3 B, const float3 C, /**/ float3 *lo, float3 *hi) {
    lo->x = min3(A.x, B.x, C.x) - BBOX_MARGIN;
    lo->y = min3(A.y, B.y, C.y) - BBOX_MARGIN;
    lo->z = min3(A.z, B.z, C.z) - BBOX_MARGIN;
    
    hi->y = max3(A.y, B.y, C.y) + BBOX_MARGIN;
    hi->x = max3(A.x, B.x, C.x) + BBOX_MARGIN;
    hi->z = max3(A.z, B.z, C.z) + BBOX_MARGIN;
}

static __device__ int encode(int soluteid, int id) {
    return soluteid * MAXT + id;
}

__global__ void countt(const int nt, const int4 *tt, const int nv, const Particle *pp, const int ns, /**/ int *counts) {
    const int thid = threadIdx.x + blockIdx.x * blockDim.x;
    float3 A, B, C, lo, hi;
    int3 t;
    int strtx, strty, strtz, endx, endy, endz;
    int iz, iy, ix, cid;
    
    if (thid >= nt * ns) return;
        
    const int tid = thid % nt;
    const int sid = thid / nt;

    t = loadt(tt, tid);

    const int base = nv * sid;

    A = loadr(pp, base + t.x);
    B = loadr(pp, base + t.y);
    C = loadr(pp, base + t.z);

    tbbox(A, B, C, /**/ &lo, &hi);

    strtx = max(int (lo.x + XS/2), 0);
    strty = max(int (lo.y + YS/2), 0);
    strtz = max(int (lo.z + ZS/2), 0);

    endx = min(int (hi.x + XS/2) + 1, XS);
    endy = min(int (hi.x + YS/2) + 1, YS);
    endz = min(int (hi.x + ZS/2) + 1, ZS);

    for (iz = strtz; iz < endz; ++iz)
    for (iy = strty; iy < endy; ++iy)
    for (ix = strtx; ix < endx; ++ix) {
        cid = ix + XS * (iy + YS * iz);
        atomicAdd(counts + cid, 1);
    }
}

__global__ void fill_ids(int soluteid, const int nt, const int4 *tt, const int nv, const Particle *pp, const int ns, const int *starts, /**/ int *counts, int *ids) {
    const int thid = threadIdx.x + blockIdx.x * blockDim.x;
    float3 A, B, C, lo, hi;
    int3 t;
    int strtx, strty, strtz, endx, endy, endz;
    int iz, iy, ix, cid;
    int subindex, start;
    
    if (thid >= nt * ns) return;
        
    const int tid = thid % nt;
    const int sid = thid / nt;

    t = loadt(tt, tid);

    const int base = nv * sid;

    A = loadr(pp, base + t.x);
    B = loadr(pp, base + t.y);
    C = loadr(pp, base + t.z);

    tbbox(A, B, C, /**/ &lo, &hi);

    strtx = max(int (lo.x + XS/2), 0);
    strty = max(int (lo.y + YS/2), 0);
    strtz = max(int (lo.z + ZS/2), 0);

    endx = min(int (hi.x + XS/2) + 1, XS);
    endy = min(int (hi.x + YS/2) + 1, YS);
    endz = min(int (hi.x + ZS/2) + 1, ZS);

    for (iz = strtz; iz < endz; ++iz)
    for (iy = strty; iy < endy; ++iy)
    for (ix = strtx; ix < endx; ++ix) {
        cid = ix + XS * (iy + YS * iz);
        subindex = atomicAdd(counts + cid, 1);
        start = starts[cid];

        ids[start + subindex] = encode(soluteid, thid);
    }
}
} // dev

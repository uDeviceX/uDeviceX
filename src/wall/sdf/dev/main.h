static __device__ float3 get_r(int3 L, const float *U) {
    enum {X, Y, Z};
    float3 r;
    r.x = L.x * (U[X] - 0.5);
    r.y = L.y * (U[Y] - 0.5);
    r.z = L.z * (U[Z] - 0.5);
    return r;
}

__global__ void count_inside(Sdf_v sv, int3 L, int n, const float *UU, int *counts) {
    int i, c, chunkid, warpmaster;
    float3 r;
    float s;

    c = 0;
    warpmaster = (0 == threadIdx.x % warpSize);
    i   = threadIdx.x + blockIdx.x * blockDim.x;
    chunkid = blockDim.x;

    if (i < n) {    
        r = get_r(L, UU + 3 * i);
        s = sdf(&sv, r.x, r.y, r.z);
        c = (s <= 0.f);
    }

    c = warpReduceSum(c);

    if (warpmaster)
        atomicAdd(&counts[chunkid], c);
}

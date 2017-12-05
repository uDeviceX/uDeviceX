static __device__ float3 warpReduceSum(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
        val.z += __shfl_down(val.z, offset);
    }
    return val;
}

__global__ void reduceByWarp(const float3 * const __restrict__ vel, const uint ncells, /**/ float3 *res) {
    assert(blockDim.x == 32);
    int i, c;
    float3 v = make_float3(0, 0, 0);
    i = threadIdx.x + blockIdx.x * blockDim.x;
    c = blockIdx.x; /* [c]hunk id */

    if (i < ncells)
        v = vel[i];
    
    v = warpReduceSum(v);

    if ((threadIdx.x % warpSize) == 0)
        res[c] = v;
}

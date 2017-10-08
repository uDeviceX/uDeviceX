static __device__ float3 warpReduceSumf3(float3 v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1) {
        v.x += __shfl_down(v.x, offset);
        v.y += __shfl_down(v.y, offset);
        v.z += __shfl_down(v.z, offset);
    }
    return v;
}

static __device__ int warpReduceSum(int v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

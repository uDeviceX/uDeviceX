template <typename T>
static __device__ T warpReduceSum(T v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

static __device__ float2 warpReduceSum(float2 val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
    }
    return val;
}

static __device__ float3 warpReduceSum(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
        val.z += __shfl_down(val.z, offset);
    }
    return val;
}

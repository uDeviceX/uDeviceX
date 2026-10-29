#ifndef UDX_WARP_COMMON_CUH
#define UDX_WARP_COMMON_CUH

template <typename T>
static __device__ T shfl_down(T val, unsigned int width) {
    return __shfl_down_sync(0xFFFFFFFF, val, width);
}

template <typename T>
static __device__ T warp_reduce_sum(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        val += shfl_down(val, offset);
    return val;
}

static __device__ float2 warp_reduce_sum(float2 val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

#endif

#ifndef UDX_RANDOM_DEVICE_COMMON_CUH
#define UDX_RANDOM_DEVICE_COMMON_CUH

namespace rnd {
static __const__ int N = 18;
static __const__ float gold = 0.6180339887498948482;
static __const__ float silver = 0.4142135623730950488;
static __const__ float bronze = 0.00008877875787352212838853023;
static __const__ float tin = 0.00004602357186447026756768986;
static __const__ float sqrt2 = 1.41421356237309514547;
__inline__ __device__ float rem(float r) { return r - floorf(r); }
__inline__ __device__ float FMA(float x, float y, float z) { return __fmaf_rz(x, y, z); }
template<int M>
__inline__ __device__ float logistic_core(float x) {
    const float x2 = x * x;
    return logistic_core<M - 2>(FMA(FMA(8.0, x2, -8.0), x2, 1.0));
}
template<> __inline__ __device__ float logistic_core<1>(float x) { return FMA(2.0 * x, x, -1.0); }
template<> __inline__ __device__ float logistic_core<0>(float x) { return x; }
__inline__ __device__ float mean0var1ii(float seed, int u, int v) {
    const float p = rem((u & 0x3FF) * gold + u * bronze + (v & 0x3FF) * silver + v * tin);
    return logistic_core<N>(seed - p) * sqrt2;
}
__inline__ __device__ float mean0var1uu(float seed, uint u, uint v) {
    const float p = rem((u & 0x3FFU) * gold + u * bronze + (v & 0x3FFU) * silver + v * tin);
    return logistic_core<N>(seed - p) * sqrt2;
}
}

#endif

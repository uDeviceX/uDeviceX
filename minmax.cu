#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"
#include "warp.cuh"

#define UDX_CONTROL_WARPSIZE_COMMON
enum { WARPSIZE = 32 };
enum { BLOCK_SIZE = 128, NWARPS = BLOCK_SIZE / WARPSIZE };

static const float minmax_minv = -100000000.;
static const float minmax_maxv =  100000000.;

static __device__ float3 fetch_pos(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 r;
    r.x = pp[i].r[X]; r.y = pp[i].r[Y]; r.z = pp[i].r[Z];
    return r;
}

static __device__ float3 minmax_minf3(float3 a, float3 b) {
    return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

static __device__ float3 minmax_maxf3(float3 a, float3 b) {
    return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

static __device__ float3 minmax_warp_reduce_min(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x = min(val.x, shfl_down(val.x, offset));
        val.y = min(val.y, shfl_down(val.y, offset));
        val.z = min(val.z, shfl_down(val.z, offset));
    }
    return val;
}

static __device__ float3 minmax_warp_reduce_max(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x = max(val.x, shfl_down(val.x, offset));
        val.y = max(val.y, shfl_down(val.y, offset));
        val.z = max(val.z, shfl_down(val.z, offset));
    }
    return val;
}

__global__ void minmax_kernel(int nv, const Particle *pp, float3 *lo, float3 *hi) {
    enum {LO, HI, D};
    int i, objid, stride, pid, wid, lid;
    float3 r, lmin, lmax;
    __shared__ float3 lohi[D][NWARPS];
    stride = blockDim.x * gridDim.x;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    lid = i % warpSize;
    wid = i / warpSize;
    objid = blockIdx.y;
    lmin = make_float3(minmax_maxv, minmax_maxv, minmax_maxv);
    lmax = make_float3(minmax_minv, minmax_minv, minmax_minv);
    for (; i < nv; i += stride) {
        pid = objid * nv + i;
        r = fetch_pos(pid, pp);
        lmin = minmax_minf3(r, lmin);
        lmax = minmax_maxf3(r, lmax);
    }
    lmin = minmax_warp_reduce_min(lmin);
    lmax = minmax_warp_reduce_max(lmax);
    if (lid == 0) {
        lohi[LO][wid] = lmin;
        lohi[HI][wid] = lmax;
    }
    __syncthreads();
    if (wid == 0) {
        if (lid < NWARPS) {
            lmin = lohi[LO][lid];
            lmax = lohi[HI][lid];
        }
        lmin = minmax_warp_reduce_min(lmin);
        lmax = minmax_warp_reduce_max(lmax);
        if (lid == 0) {
            lo[objid] = lmin;
            hi[objid] = lmax;
        }
    }
}

void minmax(const Particle *pp, int nv, int nobj, float3 *lo, float3 *hi) {
    dim3 thrd(BLOCK_SIZE, 1);
    dim3 blck(1, nobj);
    KL(minmax_kernel, (blck, thrd), (nv, pp, lo, hi));
}

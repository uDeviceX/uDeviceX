#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include "macros.cuh"
#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"
#include "kantor.cuh"

#include <math.h>

namespace cc { void check(int, const char *, int); }
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void emalloc(size_t, void **);
void efree(void *);

#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)

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

struct AreaVolume {
    int nt, nv;
    int max_cell;
    int4 *tri;
    float *av;
    float *av_hst;
    int nc;
    int Computed;
};

static __device__ float3 area_volume_position(const Particle *pp, int id) {
    enum {X, Y, Z};
    return make_float3(pp[id].r[X], pp[id].r[Y], pp[id].r[Z]);
}

static __device__ float area_volume_area(float3 r0, float3 r1, float3 r2) {
    float3 x1, x2, n;
    diff(&r1, &r0, &x1);
    diff(&r2, &r0, &x2);
    cross(&x1, &x2, &n);
    return 0.5f * sqrtf(dot<float>(&n, &n));
}

static __device__ float area_volume_volume(float3 r0, float3 r1, float3 r2) {
    return 0.1666666667f *
        ((r0.x*r1.y-r0.y*r1.x)*r2.z +
         (r0.z*r1.x-r0.x*r1.z)*r2.y +
         (r0.y*r1.z-r0.z*r1.y)*r2.x);
}

__global__ void area_volume_kernel(int nt, int nv, const Particle *pp,
                                   const int4 *tri, float *tot_av) {
    float2 av = make_float2(0.0f, 0.0f);
    int cid = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nt; i += blockDim.x * gridDim.x) {
        const int4 ids = tri[i];
        const float3 r0 = area_volume_position(pp, ids.x + cid * nv);
        const float3 r1 = area_volume_position(pp, ids.y + cid * nv);
        const float3 r2 = area_volume_position(pp, ids.z + cid * nv);
        av.x += area_volume_area(r0, r1, r2);
        av.y += area_volume_volume(r0, r1, r2);
    }
    av = warp_reduce_sum(av);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&tot_av[2 * cid + 0], av.x);
        atomicAdd(&tot_av[2 * cid + 1], av.y);
    }
}

static void area_volume_check_triangles(int nt, int nv, const int4 *tri) {
    int *rank;
    EMALLOC(nv, &rank);
    for (int i = 0; i < nv; ++i) rank[i] = 0;
    for (int i = 0; i < nt; ++i) {
        const int4 f = tri[i];
        if (f.x < 0 || f.x >= nv || f.y < 0 || f.y >= nv || f.z < 0 || f.z >= nv)
            ERR("wrong triangle: %d = [%d %d %d] (v = %d, t = %d)",
                i, f.x, f.y, f.z, nv, nt);
        rank[f.x]++;
        rank[f.y]++;
        rank[f.z]++;
    }
    for (int i = 0; i < nv; ++i)
        if (rank[i] == 0) ERR("isolated vertex: %d (v = %d, t = %d)", i, nv, nt);
    EFREE(rank);
}

void area_volume_ini(int nv, int nt, const int4 *tri, int max_cell,
                     AreaVolume **out) {
    AreaVolume *q;
    EMALLOC(1, &q);
    q->nt = nt;
    q->nv = nv;
    q->max_cell = max_cell;
    q->Computed = 0;
    Dalloc(&q->tri, nt);
    Dalloc(&q->av, 2 * max_cell);
    EMALLOC(2 * max_cell, &q->av_hst);
    UC(area_volume_check_triangles(nt, nv, tri));
    cH2D(q->tri, tri, nt);
    *out = q;
}

void area_volume_fin(AreaVolume *q) {
    Dfree(q->tri);
    Dfree(q->av);
    EFREE(q->av_hst);
    EFREE(q);
}

static void area_volume_compute_kernel(int nv, int nt, int nc,
                                       const Particle *pp, const int4 *tri,
                                       float *av) {
    dim3 threads(256, 1);
    dim3 blocks(1, nc);
    Dzero(av, 2 * nc);
    KL(area_volume_kernel, (blocks, threads), (nt, nv, pp, tri, av));
}

void area_volume_compute(AreaVolume *q, int nc, const Particle *pp, float **out) {
    if (nc > q->max_cell) ERR("nc=%d > max_cell=%d", nc, q->max_cell);
    if (!d::is_device_pointer(pp)) ERR("`pp` is not a device pointer");
    UC(area_volume_compute_kernel(q->nv, q->nt, nc, pp, q->tri, q->av));
    q->Computed = 1;
    q->nc = nc;
    *out = q->av;
}

void area_volume_host(AreaVolume *q, float **out) {
    if (!q->Computed) ERR("call `area_volume_host` before `area_volume_compute`");
    cD2H(q->av_hst, q->av, 2 * q->nc);
    *out = q->av_hst;
}

// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/force/area_volume/imp.cu */
#include <stdio.h>
#include <assert.h>

/* begin src/inc/conf.h */
/* [k]ernel [l]aunch macros */
#if !defined(KL_RELEASE)    && !defined(KL_NONE)   &&   \
    !defined(KL_SYNC)
#define KL_RELEASE
#endif

/* [c]uda [c]heck macro */
#if !defined(CC_RELEASE) && !defined(CC_SYNC)
  #define CC_RELEASE
#endif

/* who plays as device? */
#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #define DEV_CUDA
#endif

/* cuda memory allocation calls */
#if !defined(CUDA_ALLOC_RELEASE) && !defined(CUDA_ALLOC_DEBUG)
  #define CUDA_ALLOC_RELEASE
#endif

/* compute rbc force in double or float */
#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
  #define RBC_DOUBLE
#endif

/* what to do if r > lmax for a spring? */
#if !defined(RBC_SPRING_IGNORE) && !defined(RBC_SPRING_FAIL)
  #define RBC_SPRING_IGNORE
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif
/* end src/inc/conf.h */


/* begin src/inc/type.h */
template <typename T, int N>
struct Sarray { /* [s]tatic array */
    T d[N];
};

struct Particle {
    float r[3], v[3];
};

struct Rigid {
    float Iinv[6],            /* moment of inertia            6        */
        mass,                 /* mass of the solid            7        */
        com[3],               /* [c]enter [o]f [m]ass         10       */
        v[3], om[3],          /* linear and angular velocity  13 16    */
        e0[3], e1[3], e2[3],  /* local referential            19 22 25 */
        fo[3], to[3],         /* force, torque                28 31    */
        id;                   /* id of the solid              32       */
};

struct Force {
    float f[3];
};

struct Momentum {
    float P[3], L[3]; /* linear and angular momentum */
};

struct Positioncp {
    float rc[3]; /* current position  */
    float rp[3]; /* previous position */
};

typedef Sarray<int, 26>       int26;
typedef Sarray<int, 27>       int27;

typedef Sarray<int*, 26>     intp26;
typedef Sarray<Particle*, 26> Pap26;
typedef Sarray<Force*,    26> Fop26;
typedef Sarray<Momentum*, 26> Mop26;
typedef Sarray<Positioncp*, 26> Pocpp26;

/* particles wrap */
struct PaWrap {
    int n;
    const Particle *pp;
};

/* forces wrap */
struct FoWrap {
    int n;
    Force *ff;
};
/* end src/inc/type.h */
/* begin src/inc/dev.h */
/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())

#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault /* "[a]ny to [a]ny" */

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))  /* [t]to, [f]rom */
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))  /* [t]to, [f]rom */
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))

/* device allocation */
#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))

/* [d]evice set */
#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)

/* [d]evice [a]synchronous set */
#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)
/* end src/inc/dev.h */
/* begin src/math/dev.h */
/* use templates here because we might have mixed float/double */

#define _HD_ __host__ __device__

template <typename T1, typename T2>
_HD_ void scal(const T1 a, /*io*/ T2 *b) {
    b->x *= a;
    b->y *= a;
    b->z *= a;
}

template <typename T1, typename T2>
_HD_ void add(const T1 *a, /*io*/ T2 *b) {
    b->x += a->x;
    b->y += a->y;
    b->z += a->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void diff(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ T1 dot(const T2 *a, const T3 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void cross(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
_HD_ void scalmult(const T1 *a, const T2 x, /**/ const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}


template <typename T1, typename T2, typename T3, typename T4>
_HD_ void apxb(const T1 *a, const T2 x, const T3 *b, /**/ T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void axpy(const T1 a, const T2 *x, /*io*/ T3 *y) {
    y->x += a * x->x;
    y->y += a * x->y;
    y->z += a * x->z;
}
/* end src/math/dev.h */

/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */
/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

#ifndef UDX_RBC_FORCE_DEVICE_API_COMMON
#define UDX_RBC_FORCE_DEVICE_API_COMMON
/* begin src/d/api.h */
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

// tag::more[]
const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);
// end::more[]

// tag::api[]
int GetSymbolAddress(void **devPtr, const void *symbol);
int Malloc(void **p, size_t);
int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset=0, int kind=MemcpyHostToDevice);
int MemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset=0, int kind=MemcpyDeviceToHost);
int HostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
int Memcpy (void *dst, const void *src, size_t count, int kind);
int MemsetAsync (void *devPtr, int value, size_t count, Stream_t stream=0);
int Memset (void *devPtr, int value, size_t count);
int MemcpyAsync (void * dst, const void * src, size_t count, int kind, Stream_t stream = 0);
int Free (void *devPtr);
int FreeHost (void *hstPtr);
int DeviceSynchronize (void);
int PeekAtLastError(void);
// end::api[]
}
/* end src/d/api.h */
#endif
/* begin src/d/ker.h */
#if   defined(DEV_CUDA)
/* begin src/d/cuda/ker.h */
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
/* end src/d/cuda/ker.h */
#elif defined(DEV_CPU)
/* begin src/d/cpu/ker.h */
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
/* end src/d/cpu/ker.h */
/* begin src/d/cpu/wrap.h */
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
/* end src/d/cpu/wrap.h */
#else
  #error DEV_* is undefined
#endif
/* end src/d/ker.h */
/* begin src/d/q.h */
/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
/* begin src/d/cuda/q.h */
/* B.1. Function Type Qualifiers */
/* B.2. Variable Type Qualifiers */
/* end src/d/cuda/q.h */
#elif defined(DEV_CPU)
/* begin src/d/cpu/q.h */
/* B.1. Function Type Qualifiers */
#ifdef __device__
  #undef  __device__
  #define __device__
#endif

#ifdef __global__
  #undef  __global__
  #define __global__
#endif

#ifdef __host__
  #undef  __host__
  #define __host__
#endif

#ifdef __noinline__
  #undef  __noinline__
  #define __noinline__
#endif

#ifdef __forceinline__
  #undef  __forceinline__
  #define __forceinline__
#endif

/* B.2. Variable Type Qualifiers */
/* __device__ : also a function qualifier */
#ifdef __constant__
  #undef  __constant__
  #define __constant__
#endif

#ifdef __shared__
  #undef  __shared__
  #define __shared__
#endif

#ifdef __managed__
  #undef  __managed__
  #define __managed__
#endif

#ifdef __restrict__
  #undef  __restrict__
  #define __restrict__
#endif
/* end src/d/cpu/q.h */
#else
  #error DEV_* is undefined
#endif
/* end src/d/q.h */
/* begin src/utils/cc.h */
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* begin src/utils/cc/sync.h */
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
/* end src/utils/cc/sync.h */
#elif defined(CC_RELEASE)
/* begin src/utils/cc/release.h */
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
/* end src/utils/cc/release.h */
#else
  #error CC_* is undefined
#endif

/* begin src/utils/cc/common.h */
namespace cc {
void check(int rc, const char *file, int line);
}


/* end src/utils/cc/common.h */

/* TODO: header includes header */
/* end src/utils/cc.h */
/* begin src/utils/kl.h */
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
/* begin src/utils/kl/release.h */
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
/* end src/utils/kl/release.h */
#elif  defined(KL_NONE)
/* begin src/utils/kl/none.h */
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
/* end src/utils/kl/none.h */
#elif  defined(KL_SYNC)
/* begin src/utils/kl/sync.h */
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
/* end src/utils/kl/sync.h */
#else
  #error KL_* is undefined
#endif

/* begin src/utils/kl/common.h */
namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
/* end src/utils/kl/common.h */
/* begin src/utils/kl/macro.h */
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)
/* end src/utils/kl/macro.h */

/* TODO: header includes header */
/* end src/utils/kl.h */
/* begin src/algo/utils/shfl.h */
#define _I_ static __device__

#if __CUDACC_VER_MAJOR__ >= 9
template <typename T>
_I_ T shfl_down(T val, unsigned int width) {
    return __shfl_down_sync(0xFFFFFFFF, val, width);
}

template <typename T>
_I_ T shfl_up(T val, unsigned int width) {
    return __shfl_up_sync(0xFFFFFFFF, val, width);
}
template <typename T>
_I_ T shfl(T val, unsigned int width) {
    return __shfl_sync(0xFFFFFFFF, val, width);
}
#else
template <typename T>
_I_ T shfl_down(T val, unsigned int width) {
    return __shfl_down(val, width);
}

template <typename T>
_I_ T shfl_up(T val, unsigned int width) {
    return __shfl_up(val, width);
}
template <typename T>
_I_ T shfl(T val, unsigned int width) {
    return __shfl(val, width);
}
#endif

#undef _I_
/* end src/algo/utils/shfl.h */
/* begin src/algo/utils/dev.h */
#define _S_ static __device__
#define _I_ static __device__

// tag::int[]
template <typename T>
_I_ T warpReduceSum(T v)  // <1>
// end::int[]
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

// tag::int[]
_I_ float2 warpReduceSum(float2 val)  // <2>
// end::int[]
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

// tag::int[]
_I_ float3 warpReduceSum(float3 val)  // <3>
// end::int[]
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
        val.z += shfl_down(val.z, offset);
    }
    return val;
}

#undef _S_
#undef _I_

/* end src/algo/utils/dev.h */

/* begin src/rbc/force/area_volume/imp.h */
struct AreaVolume;
struct Particle;
struct int4;

// tag::interface[]
void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**); // <1>
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*, /**/ float **av); // <2>
void area_volume_host(AreaVolume*, /**/ float **av); // <3>
// end::interface[]
/* end src/rbc/force/area_volume/imp.h */

namespace rbc_force_av_dev {
/* begin src/rbc/force/area_volume/dev/main.h */
static __device__ float3 pp2r(const Particle *pp, const int id) {
    enum {X, Y, Z};
    float3 r;
    r.x = pp[id].r[X];
    r.y = pp[id].r[Y];
    r.z = pp[id].r[Z];
    return r;
}

static __device__ float area(const float3 r0, const float3 r1, const float3 r2) {
    float3 x1, x2, n;
    diff(&r1, &r0, /**/ &x1);
    diff(&r2, &r0, /**/ &x2);
    cross(&x1, &x2, /**/ &n);
    return 0.5f * sqrtf(dot<float>(&n, &n));
}

static __device__ float volume(const float3 r0, const float3 r1, const float3 r2) {
    return
        0.1666666667f *
        ((r0.x*r1.y-r0.y*r1.x)*r2.z +
         (r0.z*r1.x-r0.x*r1.z)*r2.y +
         (r0.y*r1.z-r0.z*r1.y)*r2.x);
}

__global__ void main(int nt, int nv, const Particle *pp, const int4 *tri, float *totA_V) {
    float2 a_v = make_float2(0.0f, 0.0f);
    int i, cid = blockIdx.y;
    float3 r0, r1, r2;
    int4 ids;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nt; i += blockDim.x * gridDim.x) {
        ids = tri[i];
        r0 = pp2r(pp, ids.x + cid * nv);
        r1 = pp2r(pp, ids.y + cid * nv);
        r2 = pp2r(pp, ids.z + cid * nv);

        a_v.x += area  (r0, r1, r2);
        a_v.y += volume(r0, r1, r2);
    }
    a_v = warpReduceSum(a_v);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&totA_V[2 * cid + 0], a_v.x);
        atomicAdd(&totA_V[2 * cid + 1], a_v.y);
    }
}
/* end src/rbc/force/area_volume/dev/main.h */
}
/* begin src/rbc/force/area_volume/imp/type.h */
struct AreaVolume {
    int nt, nv; /* number of triangles and vertices in one cell */
    int max_cell; /* max cell number */
    int4 *tri; /* triangles on device */
    float *av; /* output: [area0 volume0 area1 volume1, ..] */
    float *av_hst;
    int nc;       /* remember the last number of cell */
    int Computed; /* was area computed? */
};
/* end src/rbc/force/area_volume/imp/type.h */
/* begin src/rbc/force/area_volume/imp/main.h */
static void assert_tri(int nt, int nv, const int4 *tri) {
    int *rank;
    int f0, f1, f2, i;
    int4 f;
    EMALLOC(nv, &rank);
    for (i = 0; i < nv; i++) rank[i] = 0;
    for (i = 0; i < nt; i++) {
        f = tri[i];
        f0 = f.x; f1 = f.y; f2 = f.z;
        if (f0 < 0 || f0 >= nv || f1 < 0 || f1 >= nv || f2 < 0 || f2 >= nv)
            ERR("wrong triangle: %d = [%d %d %d] (v = %d, t = %d)", i, f0, f1, f2, nv, nt);
        rank[f0]++; rank[f1]++; rank[f2]++;
    }
    for (i = 0; i < nv; i++)
        if (rank[i] == 0) ERR("isolated vertex: %d (v = %d, t = %d)", i, nv, nt);
    EFREE(rank);
}

void area_volume_ini(int nv, int nt, const int4 *hst, int max_cell, /**/ AreaVolume **pq) {
    AreaVolume *q;
    EMALLOC(1, &q);
    q->nt = nt; q->nv = nv; q->max_cell = max_cell; q->Computed = 0;
    Dalloc(&q->tri, nt);
    Dalloc(&q->av , 2*max_cell);
    EMALLOC(2*max_cell, &q->av_hst);
    UC(assert_tri(nt, nv, hst));
    cH2D(q->tri, hst, nt);
    *pq = q;
}

void area_volume_fin(AreaVolume *q) {
    Dfree(q->tri);
    Dfree(q->av);
    EFREE(q->av_hst);
    EFREE(q);
}

static void compute(int nv, int nt, int nc, const Particle *pp, const int4 *tri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    Dzero(av, 2*nc);
    KL(rbc_force_av_dev::main, (avBlocks, avThreads), (nt, nv, pp, tri, av));
}

void area_volume_compute(AreaVolume *q, int nc, const Particle *pp, /**/ float **pav) {
    if (nc > q->max_cell) ERR("nc=%d > max_cell=%d", nc, q->max_cell);
    if (!d::is_device_pointer(pp)) ERR("`pp`  is not a device pointer");
    UC(compute(q->nv, q->nt, nc, pp, q->tri, q->av));
    q->Computed = 1; q->nc = nc;
    *pav = q->av;
}

void area_volume_host(AreaVolume *q, /**/ float **pav) {
    if (!q->Computed) ERR("call `area_volume_host` before `area_volume_compute`");
    cD2H(q->av_hst, q->av, 2*q->nc);
    *pav = q->av_hst;
}
/* end src/rbc/force/area_volume/imp/main.h */
/* end src/rbc/force/area_volume/imp.cu */

#line 1 "src/rbc/force/bending/imp.cpp"

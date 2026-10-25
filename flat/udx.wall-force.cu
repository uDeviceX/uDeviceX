// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/wall/force/imp.cu */
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


/* begin src/inc/def.h */
/* solvent colors */
#define XMACRO_COLOR(_)                         \
    _(BLUE)                                     \
    _(RED)

#define ENUM_COLOR(a) a##_COLOR
#define make_enum_color(a) ENUM_COLOR(a) ,
enum {
    XMACRO_COLOR(make_enum_color)
    N_COLOR
};
#undef make_enum_color

enum {
    /* wall margins */
    XWM = 6,
    YWM = 6,
    ZWM = 6,

    SAFETY_FACTOR_MAXP = 3,

    /* maximum number of particles per solid */
    MAX_PSOLID_NUM = 12000,

    /* maximum number of solids per node */
    MAX_SOLIDS = 100,

    /* maximum number of object types (solid, rbc, ...) */
    MAX_MBR_TYPES = 2,
    MAX_RIG_TYPES = 2,
    MAX_OBJ_TYPES = MAX_MBR_TYPES + MAX_RIG_TYPES,

    /* maximum number density of particles of the objects */
    MAX_OBJ_DENSITY = 30,

    /* maximum number of red blood cells per node */
    MAX_CELL_NUM = 1500,

    /* maximum texture size in bytes */
    MAX_TEXO_SIZE = 100000000,

    /* safety factor for dpd halo interactions */
    HSAFETY_FACTOR = 10,

    /* safety factor for odist fragments */
    ODSTR_FACTOR = 3,
};

/* for the spring forces r = min(lmax*RBC_SPRING_CAP, r) */
#define RBC_SPRING_CAP (0.99)
/* end src/inc/def.h */
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
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
/* begin src/utils/texo.h */
/* 1D texture object template */
static void sz_check(int n) {
    if (n <            0 ) ERR("too small texo: %d", n);
    if (n > MAX_TEXO_SIZE) ERR("too big texo: %d > %d", n, MAX_TEXO_SIZE);
}

// tag::struct[]
template<typename T>
struct Texo {
    cudaTextureObject_t d;
};
// end::struct[]

template<typename T>
// tag::int[]
void texo_setup(int n, T *data, /**/ Texo<T> *to)
// end::int[]
{
    sz_check(n);

    cudaResourceDesc resD;
    cudaTextureDesc  texD;

    memset(&resD, 0, sizeof(resD));
    resD.resType = cudaResourceTypeLinear;
    resD.res.linear.devPtr = data;
    resD.res.linear.sizeInBytes = n * sizeof(T);
    resD.res.linear.desc = cudaCreateChannelDesc<T>();

    memset(&texD, 0, sizeof(texD));
    texD.normalizedCoords = 0;
    texD.readMode = cudaReadModeElementType;

    CC(cudaCreateTextureObject(&to->d, &resD, &texD, NULL));
}

template<typename T>
// tag::int[]
void texo_destroy(/**/ Texo<T> *to)
// end::int[]
{
    CC(cudaDestroyTextureObject(to->d));
}
/* end src/utils/texo.h */
/* begin src/utils/texo.dev.h */
template<typename T>
__device__ __forceinline__
// tag::int[]
const T texo_fetch(const Texo<T> t, const int i)
// end::int[]
{
    return Tfetch(T, t.d, i);
}
/* end src/utils/texo.dev.h */

/* begin src/math/rnd/imp.h */
// tag::interface[]
struct RNDunif;                                             // <1>
void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r); // <2>
void rnd_fin(RNDunif *r);                                   // <3>
float rnd_get(RNDunif *r);                                  // <4>
// end::interface[]
/* end src/math/rnd/imp.h */
/* begin src/math/rnd/dev.h */
namespace rnd {
/* random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
   mean = 0, variance = 1 */

// passes of logistic map
static __const__ int N = 18;
// spacing coefficints for low discrepancy numbers
static __const__ float gold   = 0.6180339887498948482;
static __const__ float silver = 0.4142135623730950488;
static __const__ float bronze = 0.00008877875787352212838853023;
static __const__ float tin    = 0.00004602357186447026756768986;
static __const__ float sqrt2 = 1.41421356237309514547;

/************************* Trunk generator ***********************
 * Make one global random number per each timestep
 * cite G. Marsaglia
 * passes BigCrush
 *****************************************************************/

/************************* Branch generator **********************
 * Make one random number per pair of particles per timestep
 * Based on the Logistic map on interval [-1,1]
 * each branch no weaker than trunk
 * zero dependency between branches
 *****************************************************************/

// floating point version of LCG
__inline__ __device__ float rem( float r ) {
    return r - floorf( r );
}

// FMA wrapper for the convenience of switching rouding modes
__inline__ __device__ float FMA( float x, float y, float z ) {
    return __fmaf_rz( x, y, z );
}

// logistic rounds
// <3> : 4 FMA + 1 MUL
// <2> : 2 FMA + 1 MUL
// <1> : 1 FMA + 1 MUL

template<int N> __inline__ __device__ float __logistic_core( float x )
{
    float x2 = x * x;
    float r = FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
    return __logistic_core < N - 2 > ( r );
}

template<> __inline__ __device__ float __logistic_core<1>( float x ) {
    return FMA( 2.0 * x, x, -1.0 );
}

template<> __inline__ __device__ float __logistic_core<0>( float x ) {
    return x;
}

inline __device__ float mean0var1ii( float seed, int u, int v )
{
    float p = rem( ( ( u & 0x3FF ) * gold ) + u * bronze + ( ( v & 0x3FF ) * silver ) + v * tin ); // safe for large u or v
    float l = __logistic_core<N>( seed - p );
    return l * sqrt2;
}

inline __device__ float mean0var1uu( float seed, uint u, uint v )
{
    // 7 FLOPS
    float p = rem( ( ( u & 0x3FFU ) * gold ) + u * bronze + ( ( v & 0x3FFU ) * silver ) + v * tin ); // safe for large u or v
    // 45+1 FLOPS
    float l = __logistic_core<N>( seed - p );
    // 1 FLOP
    return l * sqrt2;
}

}
/* end src/math/rnd/dev.h */
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

/* begin src/coords/type.h */
// tag::view[]
struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
// end::view[]
/* end src/coords/type.h */
/* begin src/coords/dev.h */
/* domain sizes */

static __device__ int xdomain(const Coords_v c) {
    return c.Lx * c.xd;
}

static __device__ int ydomain(const Coords_v c) {
    return c.Ly * c.yd;
}

static __device__ int zdomain(const Coords_v c) {
    return c.Lz * c.zd;
}

/* [l]ocal to [c]enter */

static __device__ float xl2xc(const Coords_v c, float xl) {
    return xl - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yl2yc(const Coords_v c, float yl) {
    return yl - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
}

static __device__ float zl2zc(const Coords_v c, float zl) {
    return zl - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);
}

static __device__ void local2center(const Coords_v c, float3 rl, /**/ float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}

/* [c]enter to [l]ocal  */

static __device__ float xc2xl(const Coords_v c, float xc) {
    return xc + 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yc2yl(const Coords_v c, float yc) {
    return yc + 0.5f * c.Ly * (c.yd - 2.f * c.yc - 1);
}

static __device__ float zc2zl(const Coords_v c, float zc) {
    return zc + 0.5f * c.Lz * (c.zd - 2.f * c.zc - 1);
}

static __device__ void center2local(Coords_v c, float3 rc, /**/ float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}

/* [l]ocal to [g]lobal */

static __device__ float xl2xg(const Coords_v c, float xl) {
    return (c.xc + 0.5f) * c.Lx + xl;
}

static __device__ float yl2yg(const Coords_v c, float yl) {
    return (c.yc + 0.5f) * c.Ly + yl;
}

static __device__ float zl2zg(const Coords_v c, float zl) {
    return (c.zc + 0.5f) * c.Lz + zl;
}

static __device__ void local2global(const Coords_v c, float3 rl, /**/ float3 *rg) {
    rg->x = xl2xg(c, rl.x);
    rg->y = yl2yg(c, rl.y);
    rg->z = zl2zg(c, rl.z);
}

/* [g]lobal to [l]ocal */

static __device__ float xg2xl(const Coords_v c, float xg) {
    return xg - (c.xc + 0.5f) * c.Lx;
}

static __device__ float yg2yl(const Coords_v c, float yg) {
    return yg - (c.yc + 0.5f) * c.Ly;
}

static __device__ float zg2zl(const Coords_v c, float zg) {
    return zg - (c.zc + 0.5f) * c.Lz;
}

static __device__ void global2local(const Coords_v c, float3 rg, /**/ float3 *rl) {
    rl->x = xg2xl(c, rg.x);
    rl->y = yg2yl(c, rg.y);
    rl->z = zg2zl(c, rg.z);
}

/* end src/coords/dev.h */
/* begin src/coords/imp.h */
struct Coords;
struct Coords_v;
struct float3;
struct int3;

// tag::view[]
void coords_get_view(const Coords *c, Coords_v *v);
// end::view[]

// tag::domainsz[]
int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);
// end::domainsz[]

// tag::local2center[]
float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);
// end::local2center[]

// tag::center2local[]
float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);
// end::center2local[]

// tag::local2global[]
float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);
// end::local2global[]

// tag::global2local[]
float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);
// end::global2local[]

/* edges of the sub-domain in global coordinates */

// tag::bblocal[]
int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);
// end::bblocal[]

// tag::subdomainsz[]
int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);
// end::subdomainsz[]

// tag::int[]
/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);
// end::int[]
/* end src/coords/imp.h */

/* begin src/wall/wvel/type.h */
/* view structures                      */
/* (what is passed to device functions) */

struct WvelCste_v {
    float3 u;
};

struct WvelShear_v {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // only half is moving ?
};

struct WvelHS_v {
    float u; // radial max inflow
    float h; // height of the channel
};

/* end src/wall/wvel/type.h */
/* begin src/wall/wvel/imp.h */
struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

// tag::enum[]
enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};
// end::enum[]

// tag::mem[]
void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);

void wvel_step_ini(WvelStep **wv);
void wvel_step_fin(WvelStep *wv);
// end::mem[]

// tag::ini[]
void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
void wvel_set_timestep(float dt, Wvel *vw);
// end::ini[]

// tag::cnf[]
void wvel_set_conf(const Config *cfg, Wvel *vw);
// end::cnf[]

// tag::step[]
void wvel_get_step(float t, const Wvel *wv, /**/ WvelStep *);
// end::step[]

// tag::get[]
int wvel_get_type(const WvelStep *w);
// end::get[]

// tag::view[]
void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v);
// end::view[]
/* end src/wall/wvel/imp.h */
/* begin src/wall/wvel/dev.h */
// tag::vel[]
static __device__ void wvel(WvelCste_v p, Coords_v c, float3 r, /**/ float3 *v)
// end::vel[]
{
    *v = p.u;
}

// tag::vel[]
static __device__ void wvel(WvelShear_v p, Coords_v c, float3 r, /**/ float3 *v)
// end::vel[]
{
    float3 rc; // relative to center
    float gdot, d;
    int vdir, gdir;

    gdot = p.gdot;
    vdir = p.vdir;
    gdir = p.gdir;

    local2center(c, r, /**/ &rc);

    d = 0;
    if      (gdir == 0) d = rc.x;
    else if (gdir == 1) d = rc.y;
    else if (gdir == 2) d = rc.z;

    d = p.half ? min(d, 0.f) : d;

    v->x = v->y = v->z = 0;
    if      (vdir == 0) v->x = d * gdot;
    else if (vdir == 1) v->y = d * gdot;
    else if (vdir == 2) v->z = d * gdot;
}

// tag::vel[]
static __device__ void wvel(WvelHS_v p, Coords_v c, float3 r, /**/ float3 *v)
// end::vel[]
{
    float3 rc; // relative to center
    float u, h, r2inv, hfac;

    u = p.u;
    h = p.h;

    local2center(c, r, /**/ &rc);

    r2inv = 1.f / (rc.x*rc.x + rc.y*rc.y);
    hfac = u * (1.f - rc.z * rc.z * 4 / (h*h));
    hfac = max(0.f, hfac);

    v->x = r2inv * hfac * rc.x;
    v->y = r2inv * hfac * rc.y;
    v->z = 0;
}

// tag::bb[]
template <typename Wvel_v>
static __device__ void bounce_vel(Wvel_v wv, Coords_v c, float3 rw, /* io */ float3* v)
// end::bb[]
{
    float3 vw;
    wvel(wv, c, rw, /**/ &vw);
    v->x = 2 * vw.x - v->x;
    v->y = 2 * vw.y - v->y;
    v->z = 2 * vw.z - v->z;
}
/* end src/wall/wvel/dev.h */

/* begin src/math/tform/type.h */
struct Tform_v { float o[3], s[3]; };
/* end src/math/tform/type.h */
/* begin src/math/tform/dev.h */
static __device__ void tform_convert_dev(const Tform_v *t, const float a0[3], /**/ float a1[3]) {
    enum {X, Y, Z};
    const float *o, *s;
    o = t->o; s = t->s;
    a1[X] = s[X]*a0[X] + o[X];
    a1[Y] = s[Y]*a0[Y] + o[Y];
    a1[Z] = s[Z]*a0[Z] + o[Z];
}

static __device__ void tform_spacing_dev(const Tform_v *t, /**/ float s[3]) {
    enum {X, Y, Z};
    s[X] = t->s[X];
    s[Y] = t->s[Y];
    s[Z] = t->s[Z];
}
/* end src/math/tform/dev.h */

/* begin src/wall/sdf/tex3d/type.h */
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
/* end src/wall/sdf/tex3d/type.h */
/* begin src/wall/sdf/type.h */
/* view */
struct Sdf_v {
    Tex3d_v tex;
    Tform_v t;
    float cheap_threshold;
};
/* end src/wall/sdf/type.h */
/* begin src/wall/sdf/dev.h */
#define _S_ static __device__
#define _I_ static __device__

_S_ float fetch(const Sdf_v *sdf, float i, float j, float k) {
    return Ttex3D(float, sdf->tex.t, i, j, k);
}

_S_ int iround(float x) { return (x > 0.5) ? (x + 0.5) : (x - 0.5); }

_S_ void convert(const Sdf_v *sdf, const float a[3], /**/ float b[3]) {
    tform_convert_dev(&sdf->t, a, /**/ b);
}

_S_ void convert_floor(const Sdf_v *sdf, const float a[3], /**/ int i[3]) {
    enum {X, Y, Z};
    float f[3];
    convert(sdf, a, /**/ f);
    i[X] = int(f[X]); i[Y] = int(f[Y]); i[Z] = int(f[Z]);
}

_S_ void convert_round(const Sdf_v *sdf, const float a[3], /**/ int i[3]) {
    enum {X, Y, Z};
    float f[3];
    convert(sdf, a, /**/ f);
    i[X] = iround(f[X]); i[Y] = iround(f[Y]); i[Z] = iround(f[Z]);
}

_S_ void spacing(const Sdf_v *sdf, float s[3]) {
    tform_spacing_dev(&sdf->t, /**/ s);
}

_S_ float3 grad(const Sdf_v *sdf, const float3 *pos) {
    int tc[3];
    float fcts[3], r[3] = {pos->x, pos->y, pos->z};
    float myval, gx, gy, gz;
    convert_floor(sdf, r, /**/ tc);
    spacing(sdf, /**/ fcts);

#define tex0(ix, iy, iz) (fetch(sdf, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    myval = tex0(0, 0, 0);
    gx = fcts[0] * (tex0(1, 0, 0) - myval);
    gy = fcts[1] * (tex0(0, 1, 0) - myval);
    gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0
    return make_float3(gx, gy, gz);
}

// tag::int[]
_I_ float3 sdf_ugrad(const Sdf_v *texsdf, const float3 *r) // <1>
// end::int[]
{
    float mag, eps;
    float3 g;
    eps = 1e-6;
    g = grad(texsdf, r);
    mag = sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
    if (mag > eps) {
        g.x /= mag;
        g.y /= mag;
        g.z /= mag;
    }
    return g;
}

_S_ float cheap_sdf(const Sdf_v *sdf, float x, float y, float z)  {
    int tc[3];
    float r[3] = {x, y, z};
    convert_round(sdf, r, /**/ tc);
    return fetch(sdf, tc[0], tc[1], tc[2]);
}

// tag::int[]
_I_ bool sdf_far(const Sdf_v *sdf, float x, float y, float z) // <2>
// end::int[]
{
    return cheap_sdf(sdf, x, y, z) <= sdf->cheap_threshold - 1 ;
}

// tag::int[]
_I_ float sdf_eval(const Sdf_v *sdf0, float x, float y, float z)  // <3>
// end::int[]
{
    int c;
    float s000, s001, s010, s100, s101, s011, s110, s111;
    float s00x, s01x, s10x, s11x;
    float s0yx, s1yx;
    float szyx;
    float tc0[3], tc[3], lmbd[3], r[3] = {x, y, z};
    convert(sdf0, r, /**/ tc0);
    for (c = 0; c < 3; ++c) {
        lmbd[c] = tc0[c] - (int)tc0[c];
        tc[c] = (int)tc0[c] + 0.5;
    }
#define tex0(ix, iy, iz) (fetch(sdf0, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    s000 = tex0(0, 0, 0), s001 = tex0(1, 0, 0), s010 = tex0(0, 1, 0);
    s011 = tex0(1, 1, 0), s100 = tex0(0, 0, 1), s101 = tex0(1, 0, 1);
    s110 = tex0(0, 1, 1), s111 = tex0(1, 1, 1);
#undef tex0

#define wavrg(A, B, p) A*(1-p) + p*B /* weighted average */
    s00x = wavrg(s000, s001, lmbd[0]);
    s01x = wavrg(s010, s011, lmbd[0]);
    s10x = wavrg(s100, s101, lmbd[0]);
    s11x = wavrg(s110, s111, lmbd[0]);

    s0yx = wavrg(s00x, s01x, lmbd[1]);
    s1yx = wavrg(s10x, s11x, lmbd[1]);

    szyx = wavrg(s0yx, s1yx, lmbd[2]);
#undef wavrg
    return szyx;
}

#undef _S_
#undef _I_
/* end src/wall/sdf/dev.h */

/* begin src/pair/type.h */
/* maximum of pairwise dpd parameters */
enum {
    MAX_PAR = (N_COLOR * (N_COLOR + 1)) / 2
};


struct PairDPD {
    float a, g, s, spow;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], spow;
};

/* mirrored */
struct PairDPDCM {
    int ncolors;
    float a[N_COLOR], g[N_COLOR], s[N_COLOR], spow;
};

struct PairDPDLJ {
    float a, g, s, spow;
    float ljs, lje;
};

struct PairAdhesion {
    float k1, k2;
};

struct PairPa {
    float x, y, z;
    float vx, vy, vz;
    /* optional fields */
    int color;
};

struct PairFo {
    float x, y, z;
};

struct PairSFo {
    float x, y, z;
    /* stress */
    float sxx, syy, szz, sxy, sxz, syz;
};
/* end src/pair/type.h */
/* begin src/pair/dev.h */
#define _S_ static __device__
#define _I_ static __device__

_S_ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}

static const float EPS = 1e-6;
enum {NORM_OK, NORM_BIG, NORM_SMALL};
_S_ int norm(/*io*/ float3 *pos, /**/ float *pr, float *pinvr) {
    /* normalize r = [x, y, z], sets |r| and 1/|r| if not big */
    float x, y, z, invr, r, r2;
    x = pos->x; y = pos->y; z = pos->z;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return NORM_BIG;
    else if (r2 < EPS) {
        *pr = pos->x = pos->y = pos->z = 0;
        return NORM_SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        pos->x = x; pos->y = y; pos->z = z; *pr = r; *pinvr = invr;
        return NORM_OK;
    }
}

_S_ float ker_wrf(float s, float x) {
    return powf(x, s);
}

_S_ float magn_dpd(float a, float g, float s, float kpow,
                   float rnd, float r, float ev) {
    float wr, wc;
    float rm, f0;

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = ker_wrf(kpow, rm);

    f0  = (-g * wr * ev + s * rnd) * wr;
    f0 +=                        a * wc;
    return f0;
}

_S_ float magn_lj(float s, float e, float invr) {
    float t2, t4, t6, f;
    t2 = s * s * invr * invr;
    t4 = t2 * t2;
    t6 = t4 * t2;
    f = e * 24 * invr * t6 * (2 * t6 - 1);
    f = cap(f, 0, 1e4);
    return f;
}

_S_ float magn_adhesion(float k1, float k2, float r, float ev) {
    float spring, visc;
    spring = - k1 * r;
    visc   = - k2 * ev;
    return spring + visc;
}

_S_ float force_magn(const PairDPD *p, float rnd, float ev, float r, float) {
    return magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
}

_S_ float force_magn(const PairDPDLJ *p, float rnd, float ev, float r, float invr) {
    float f;
    f  = magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
    f += magn_lj(p->ljs, p->lje, invr);
    return f;
}

_S_ float force_magn(const PairAdhesion *p, float, float ev, float r, float) {
    return magn_adhesion(p->k1, p->k2, r, ev);
}

_S_ void magn2fo(float f0, float3 er, float3 dr, /**/ PairFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;
}

_S_ void magn2fo(float f0, float3 er, float3 dr, /**/ PairSFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;

    f->sxx = 0.5f * f->x * dr.x;
    f->sxy = 0.5f * f->x * dr.y;
    f->sxz = 0.5f * f->x * dr.z;
    f->syy = 0.5f * f->y * dr.y;
    f->syz = 0.5f * f->y * dr.z;
    f->szz = 0.5f * f->z * dr.z;
}

_S_ void make_zero(PairFo *f) {
    f->x = f->y = f->z = 0;
}

_S_ void make_zero(PairSFo *f) {
    f->x = f->y = f->z = 0;
    f->sxx = f->sxy = f->sxz = 0;
    f->syy = f->syz = f->szz = 0;
}

// tag::int[]
template <typename Param, typename Fo>
_I_ void pair_force(const Param *p, PairPa a, PairPa b, float rnd, /**/ Fo *f)
// end::int[]
{
    float r, invr, ev, f0;
    float3 dr, er, dv;
    int vnstat; /* vector normalization status */

    dr.x = a.x - b.x;
    dr.y = a.y - b.y;
    dr.z = a.z - b.z;

    dv.x = a.vx - b.vx;
    dv.y = a.vy - b.vy;
    dv.z = a.vz - b.vz;

    er = dr;

    vnstat = norm(/*io*/ &er, /*o*/ &r, &invr);
    if (vnstat == NORM_BIG) {
        make_zero(f);
        return;
    }

    ev = dot<float>(&er, &dv);

    f0 = force_magn(p, rnd, ev, r, invr);
    magn2fo(f0, er, dr, /**/ f);
}

_S_ int colors2pid(int ca, int cb) {
    int c0, c1;
    c0 = ca < cb ? ca : cb;
    c1 = ca < cb ? cb : ca;
    return c1 * (c1+1) / 2 + c0;
}

template <typename Fo>
_I_ void pair_force(const PairDPDC *pc, PairPa a, PairPa b, float rnd, /**/ Fo *f) {
    PairDPD p;
    int pid;
    pid = colors2pid(a.color, b.color);
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd, /**/ f);
}

/* mirrored: parameters from particle "a" only */
template <typename Fo>
_I_ void pair_force(const PairDPDCM *pc, PairPa a, PairPa b, float rnd, /**/ Fo *f) {
    PairDPD p;
    int pid = a.color;
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd, /**/ f);
}


// tag::add[]
_I_ void pair_add(const PairFo *b, /**/ PairFo *a)
// end::add[]
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;
}

// tag::add[]
_I_ void pair_add(const PairSFo *b, /**/ PairSFo *a)
// end::add[]
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;

    a->sxx += b->sxx;
    a->sxy += b->sxy;
    a->sxz += b->sxz;
    a->syy += b->syy;
    a->syz += b->syz;
    a->szz += b->szz;
}

#undef _S_
#undef _I_
/* end src/pair/dev.h */
/* begin src/pair/imp.h */
struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

// tag::mem[]
void pair_ini(PairParams **);
void pair_fin(PairParams *);
// end::mem[]

// tag::set[]
void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p); // <1>
void pair_set_lj(float sigma, float eps, PairParams *p); // <2>
// end::set[]
void pair_set_adhesion(float k1, float k2, PairParams *p);

// tag::cnf[]
void pair_set_conf(const Config *, const char *name_space, PairParams *);
// end::cnf[]

// tag::upd[]
void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);
// end::upd[]

// tag::view[]
void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
// end::view[]
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);
/* end src/pair/imp.h */

/* begin src/struct/parray/type.h */
struct PaArray_v {
    const float *pp;
};

struct PaCArray_v {
    const float *pp;
    const int *cc;   /* colors */
};
/* end src/struct/parray/type.h */
/* begin src/struct/parray/imp.h */
struct PaArray {
    bool colors;
    const float *pp;
    const int *cc;
};

struct PaArray_v;
struct PaCArray_v;

struct Particle;

// tag::push[]
void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);
// end::push[]

// tag::get[]
bool parray_is_colored(const PaArray *a);
// end::get[]

// tag::view[]
void parray_get_view(const PaArray *a, PaArray_v *v);  // <1>
void parray_get_view(const PaArray *a, PaCArray_v *v); // <2>
// end::view[]
/* end src/struct/parray/imp.h */
/* begin src/struct/parray/dev.h */
#define _S_ static __device__
#define _I_ static __device__

template<typename PA>
_S_ void parray_get_pp(PA a, int i, /**/ PairPa *p) {
    const float2 *pp = (const float2 *) a.pp;
    float2 s0, s1, s2;
    s0 = __ldg(pp + 3*i + 0);
    s1 = __ldg(pp + 3*i + 1);
    s2 = __ldg(pp + 3*i + 2);

    p->x = s0.x;
    p->y = s0.y;
    p->z = s1.x;

    p->vx = s1.y;
    p->vy = s2.x;
    p->vz = s2.y;
}

// tag::int[]
_I_ void parray_get(PaArray_v a, int i, /**/ PairPa *p) // <1>
// end::int[]
{
    parray_get_pp(a, i, /**/ p);
}

// tag::int[]
_I_ void parray_get(PaCArray_v a, int i, /**/ PairPa *p) // <2>
// end::int[]
{
    parray_get_pp(a, i, /**/ p);
    p->color = a.cc[i];
}

#undef _S_
#undef _I_
/* end src/struct/parray/dev.h */

/* begin src/struct/farray/type.h */
struct FoArray_v {
    float *ff;
};

struct FoSArray_v {
    float *ff;
    float *ss;
};
/* end src/struct/farray/type.h */
/* begin src/struct/farray/imp.h */
struct FoArray {
    bool stress;
    float *ff;
    float *ss; /* stress */
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

// tag::push[]
void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);
// end::push[]

void farray_clear(FoArray *);

// tag::get[]
bool farray_has_stress(const FoArray *a);
// end::get[]

// tag::view[]
void farray_get_view(const FoArray *a, FoArray_v *v);  // <1>
void farray_get_view(const FoArray *a, FoSArray_v *v); // <2>
// end::view[]
/* end src/struct/farray/imp.h */
/* begin src/struct/farray/dev.h */
#define _I_ static __device__
#define _S_ static __device__

template <int S, typename Fo, typename FA>
_S_ void farray_add_force(const Fo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    af[X] += S * f->x;
    af[Y] += S * f->y;
    af[Z] += S * f->z;
}

_S_ void farray_add_stress(const PairSFo *f, int i, /**/ FoSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    s[XX] += f->sxx;
    s[XY] += f->sxy;
    s[XZ] += f->sxz;
    s[YY] += f->syy;
    s[YZ] += f->syz;
    s[ZZ] += f->szz;
}

// tag::add[]
template <int S>
_I_ void farray_add(const PairFo *f, int i, /**/ FoArray_v a) // <1>
// end::add[]
{
    farray_add_force<S>(f, i, /**/ a);
}

// tag::add[]
template <int S>
_I_ void farray_add(const PairSFo *f, int i, /**/ FoSArray_v a) // <2>
// end::add[]
{
    farray_add_force<S>(f, i, /**/ a);
    farray_add_stress(f, i, /**/ a);
}


template <int S, typename Fo, typename FA>
_S_ void farray_atomic_add_force(const Fo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    atomicAdd(&af[X], S * f->x);
    atomicAdd(&af[Y], S * f->y);
    atomicAdd(&af[Z], S * f->z);
}

_S_ void farray_atomic_add_stress(const PairSFo *f, int i, /**/ FoSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    atomicAdd(&s[XX], f->sxx);
    atomicAdd(&s[XY], f->sxy);
    atomicAdd(&s[XZ], f->sxz);
    atomicAdd(&s[YY], f->syy);
    atomicAdd(&s[YZ], f->syz);
    atomicAdd(&s[ZZ], f->szz);
}

// tag::atomic[]
template <int S>
_I_ void farray_atomic_add(const PairFo *f, int i, /**/ FoArray_v a)
// end::atomic[]
{
    farray_atomic_add_force<S>(f, i, /**/ a);
}

// tag::atomic[]
template <int S>
_I_ void farray_atomic_add(const PairSFo *f, int i, /**/ FoSArray_v a)
// end::atomic[]
{
    farray_atomic_add_force<S>(f, i, /**/ a);
    farray_atomic_add_stress(f, i, /**/ a);
}


// tag::ini[]
_I_ PairFo farray_fo0(FoArray_v)
// end::ini[]
{
    PairFo f;
    f.x = f.y = f.z = 0;
    return f;
}

// tag::ini[]
_I_ PairSFo farray_fo0(FoSArray_v)
// end::ini[]
{
    PairSFo f;
    f.x = f.y = f.z = 0;
    f.sxx = f.sxy = f.sxz = 0;
    f.syy = f.syz = f.szz = 0;
    return f;
}

#undef _I_
#undef _S_
/* end src/struct/farray/dev.h */

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
/* begin src/wall/force/imp.h */
struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct RNDunif;

// tag::struct[]
struct WallForce { /* local wall data */
    Sdf_v sdf_v;
    Texo<int> start;
    Texo<float4> pp;
    int n;
    int3 L;
};
// end::struct[]

struct WallRepulsePrm {
    float l;
};

// tag::int[]
void wall_force_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                      /**/ const FoArray*); // <1>

void wall_force_adhesion_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                               /**/ const FoArray*);

void wall_force_repulse(Sdf_v, WallRepulsePrm, long n, const PaArray*, const FoArray*); // <2>
// end::int[]
/* end src/wall/force/imp.h */

namespace wf_dev {
namespace map {
/* begin src/wall/force/dev/map/type.h */
struct Map { /* helps to find remote particle */
    int org0, org1, org2;
    int str0, str1, str2;
};
/* end src/wall/force/dev/map/type.h */
/* begin src/wall/force/dev/map/ini.h */
static __device__ int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }

static __device__ void ini(int3 L, int zplane, const Texo<int> texstart, int w_n, float x, float y, float z, /**/ Map *m) {
#define start_fetch(i) (texo_fetch(texstart, i))
    int str0, str1, str2;
    int org0, org1, org2;
    int xcells, ycells, zcells, ncells;
    int xbase, ybase, zbase;
    int cid0, cid1, cid2;
    int count0, count1, count2;

    xcells = L.x + 2 * XWM;
    ycells = L.y + 2 * YWM;
    zcells = L.z + 2 * ZWM;

    ncells = xcells * ycells * zcells;

    xbase = (int)(x + xcells / 2);
    ybase = (int)(y + ycells / 2);
    zbase = (int)(z + zcells / 2);

    xbase = minmax(1, xcells - 2, xbase);
    ybase = minmax(1, ycells - 2, ybase);
    zbase = minmax(1, zcells - 2, zbase);

    cid0 = xbase - 1 + xcells * (ybase - 1 + ycells * (zbase - 1 + zplane));
    org0 = start_fetch(cid0);
    count0 = start_fetch(cid0 + 3) - org0;

    cid1 = cid0 + xcells;
    org1 = start_fetch(cid1);
    count1 = start_fetch(cid1 + 3) - org1;

    cid2 = cid0 + xcells * 2;
    org2 = start_fetch(cid2);
    count2 = cid2 + 3 == ncells
        ? w_n
        : start_fetch(cid2 + 3) - org2;

    str0 = count0;
    str1 = count0 + count1;
    str2 = str1 + count2;

    org1 -= str0;
    org2 -= str1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->str0 = str0; m->str1 = str1; m->str2 = str2;
#undef start_fetch
}
/* end src/wall/force/dev/map/ini.h */
/* begin src/wall/force/dev/map/use.h */
/* true if `i' is bigger than the number of remote particles */
static __device__ int endp(const Map m, int i) { return i >= m.str2; }
static __device__ int m2id(const Map m, int i) {
    /* return remote particle id */
    int m1, m2, id;
    m1 = (i >= m.str0);
    m2 = (i >= m.str1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}
/* end src/wall/force/dev/map/use.h */
}
/* begin src/wall/force/dev/main.h */
template <typename Wvel_v>
static __device__ void fetch_wall(Wvel_v wv, Coords_v c, Texo<float4> pp, int i, /**/ PairPa *a) {
    float3 r, v; /* wall velocity */
    float4 r0;
    r0 = texo_fetch(pp, i);
    r = make_float3(r0.x, r0.y, r0.z);
    wvel(wv, c, r, /**/ &v);

    a->x  = r.x;  a->y  = r.y;   a->z  = r.z;
    a->vx = v.x;  a->vy = v.y;   a->vz = v.z;
}

template <typename Par, typename Wvel_v, typename Fo>
static __device__ void force0(Par params, Wvel_v wv, Coords_v c, PairPa a, int aid, int zplane,
                              float seed, WallForce wa, /**/ Fo *fa) {
    map::Map m;
    PairPa b;  /* wall particles */
    Fo f;
    float rnd;
    int i, bid;

    if (sdf_far(&wa.sdf_v, a.x, a.y, a.z)) return;

    map::ini(wa.L, zplane, wa.start, wa.n, a.x, a.y, a.z, /**/ &m);

    for (i = 0; !map::endp(m, i); ++i) {
        bid = map::m2id(m, i);
        fetch_wall(wv, c, wa.pp, bid, /**/ &b);
        rnd = rnd::mean0var1ii(seed, aid, bid);
        pair_force(&params, a, b, rnd, /**/ &f);
        pair_add(&f, fa);
    }
}

template <typename Par, typename Wvel_v, typename Parray, typename Farray>
__global__ void force(Par params, Wvel_v wv, Coords_v c, Parray parray, int np, float seed, WallForce wa, /**/ Farray farray) {
    PairPa a; /* bulk particle */
    int gid, aid, zplane;
    auto ftot = farray_fo0(farray);
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid    = gid / 3;
    zplane = gid % 3;

    if (aid >= np) return;
    parray_get(parray, aid, /**/ &a);

    force0(params, wv, c, a, aid, zplane, seed, wa, /**/ &ftot);

    farray_atomic_add<1>(&ftot, aid, /**/ farray);
}

static __device__ float rep_kernel(const WallRepulsePrm *wr, float x) {
    float arg, val;
    arg = wr->l * (x + 1.f);
    val = expf(arg) - 1.f;
    return min(1e4f, val);
}

static __device__ void repulse_one_p(const WallRepulsePrm *wr, const float3 *r, const Sdf_v *sdf_v, float3 *f) {
    float s, f0;
    float3 g;
    s = sdf_eval(sdf_v, r->x, r->y, r->z);
    g = sdf_ugrad(sdf_v, r);
    f0 = - rep_kernel(wr, s);
    f->x = f0 * g.x;
    f->y = f0 * g.y;
    f->z = f0 * g.z;
}

static __device__ float3 get_pos(const Particle *p) {
    enum {X, Y, Z};
    return make_float3(p->r[X], p->r[Y], p->r[Z]);
}

__global__ void repulse(WallRepulsePrm wr, int n, const Particle *pp, Sdf_v sdf_v, /**/ Force *ff) {
    enum {X, Y, Z};
    int pid;
    float3 r, f;
    pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= n) return;
    r = get_pos(pp + pid);

    if (sdf_far(&sdf_v, r.x, r.y, r.z)) return;

    repulse_one_p(&wr, &r, &sdf_v, &f);

    ff[pid].f[X] += f.x;
    ff[pid].f[Y] += f.y;
    ff[pid].f[Z] += f.z;
}
/* end src/wall/force/dev/main.h */
}

/* begin src/wall/force/imp/main.h */
enum {
    TPP = 3 /* # threads per particle */
};

template <typename Par, typename Wvel_v, typename Parray, typename Farray>
static void interactions(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ Farray farray) {
    Coords_v coordsv;
    coords_get_view(c, &coordsv);

    KL(wf_dev::force,
       (k_cnf(TPP*n)),
       (params, wv, coordsv, parray, n, rnd_get(rnd), wa, /**/ farray));
}

template <typename Par, typename Wvel_v, typename Parray>
static void stress_dispatch(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {
    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, wv, c, parray, n, rnd, wa, /**/ farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, wv, c, parray, n, rnd, wa, /**/ farray_v);
    }
}

template <typename Par, typename Parray>
static void wvel_dispatch(Par params, const WvelStep *w, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {
    switch (wvel_get_type(w)) {
    case WALL_VEL_V_CSTE: {
        WvelCste_v wv;
        wvel_get_view(w, /**/ &wv);
        stress_dispatch(params, wv, c, parray, n, rnd, wa, /**/ farray);
        break;
    }
    case WALL_VEL_V_SHEAR: {
        WvelShear_v wv;
        wvel_get_view(w, /**/ &wv);
        stress_dispatch(params, wv, c, parray, n, rnd, wa, /**/ farray);
        break;
    }
    case WALL_VEL_V_HS: {
        WvelHS_v wv;
        wvel_get_view(w, /**/ &wv);
        stress_dispatch(params, wv, c, parray, n, rnd, wa, /**/ farray);
        break;
    }
    };
}

void wall_force_apply(const PairParams *params, const WvelStep *wv, const Coords *c, const PaArray *parray, int n,
                      RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {
    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaArray_v pav;
        UC(pair_get_view_dpd_mirrored(params, &pv));
        parray_get_view(parray, &pav);
        wvel_dispatch(pv, wv, c, pav, n, rnd, wa, /**/ farray);
    }
    else {
        PairDPD pv;
        PaArray_v pav;
        UC(pair_get_view_dpd(params, &pv));
        parray_get_view(parray, &pav);
        wvel_dispatch(pv, wv, c, pav, n, rnd, wa, /**/ farray);
    }
}

void wall_force_adhesion_apply(const PairParams *params, const WvelStep *wv, const Coords *c, const PaArray *parray, int n,
                               RNDunif *rnd, WallForce wa, /**/ const FoArray *farray) {
    PairAdhesion pv;
    PaArray_v pav;

    UC(pair_get_view_adhesion(params, &pv));
    UC(parray_get_view(parray, &pav));
    UC(wvel_dispatch(pv, wv, c, pav, n, rnd, wa, /**/ farray));
}


void wall_force_repulse(Sdf_v sdf_v, WallRepulsePrm wr, long n, const PaArray *pa, const FoArray *fo) {
    KL(wf_dev::repulse,
       (k_cnf(n)),
       (wr, n, (const Particle*) pa->pp, sdf_v, (Force*) fo->ff));
}
/* end src/wall/force/imp/main.h */
/* end src/wall/force/imp.cu */

#line 1 "src/wall/imp.cu"

// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/fluforces/bulk/imp.cu */
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
#ifndef UDX_FLUFORCES_TYPE_COMMON
#define UDX_FLUFORCES_TYPE_COMMON
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
#endif
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

/* begin src/d/api.h */
#ifndef UDX_FLUFORCES_DEVICE_API_COMMON
#define UDX_FLUFORCES_DEVICE_API_COMMON
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
/* begin src/utils/kl.h */
#ifndef UDX_FLUFORCES_KL_COMMON
#define UDX_FLUFORCES_KL_COMMON
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
#endif
/* begin src/utils/cc.h */
#ifndef UDX_FLUFORCES_CC_COMMON
#define UDX_FLUFORCES_CC_COMMON
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
#endif
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

/* begin src/struct/farray/type.h */
struct FoArray_v {
    float *ff;
};

struct FoSArray_v {
    float *ff;
    float *ss;
};
/* end src/struct/farray/type.h */
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

/* begin src/fluforces/bulk/imp.h */
struct PairParams;
struct Force;
struct float4;
struct int3;
struct RNDunif;
struct FoArray;

#ifndef UDX_FLUFORCES_BPA_COMMON
#define UDX_FLUFORCES_BPA_COMMON
struct BPaArray {
    bool colors;
    const float4 *pp;
    const int *cc;
};
#endif

void flocal_push_pp(const float4 *pp, BPaArray *a);
void flocal_push_cc(const int *cc, BPaArray *a);

void flocal_apply(const PairParams*, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd,
                  /**/ const FoArray *ff);

/* end src/fluforces/bulk/imp.h */
/* begin src/fluforces/bulk/imp/type.h */
struct TBPaArray_v {
    Texo<float4> pp;
};

struct TBPaCArray_v {
    Texo<float4> pp;
    Texo<int> cc;
};
/* end src/fluforces/bulk/imp/type.h */

namespace fbulk_dev {
/* begin src/fluforces/bulk/dev/fetch.h */
#define _S_ static __device__
#define _I_ static __device__

/* bulk parray fetch */

template <typename Parray>
_S_ void bpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = a.pp[2*i + 0];
    v = a.pp[2*i + 1];

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}


/* texture bulk parray fetch */

template <typename Parray>
_S_ void tbpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = texo_fetch(a.pp, 2*i + 0);
    v = texo_fetch(a.pp, 2*i + 1);

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}

_I_ void fetch(TBPaArray_v a, int i, PairPa *p) {
    tbpa_fetch_p(a, i, p);
}

_I_ void fetch(TBPaCArray_v a, int i, PairPa *p) {
    tbpa_fetch_p(a, i, p);
    p->color = texo_fetch(a.cc, i);
}

#undef _S_
#undef _I_
/* end src/fluforces/bulk/dev/fetch.h */
/* begin src/fluforces/bulk/dev/main.h */
#define _S_ static __device__

_S_ bool cutoff_range(PairPa pa, PairPa pb) {
    float x, y, z;
    x = pa.x - pb.x;
    y = pa.y - pb.y;
    z = pa.z - pb.z;
    return x*x + y*y + z*z <= 1.f;
}

_S_ int3 get_cid(int3 L, const PairPa *pa) {
    int3 c;
    c.x = pa->x + L.x / 2;
    c.y = pa->y + L.y / 2;
    c.z = pa->z + L.z / 2;
    return c;
}

_S_ bool valid_c(int c, int hi) {
    return (c >= 0) && (c < hi);
}

template <typename Par, typename Parray, typename Farray, typename Fo>
_S_ void loop_pp(const Par *params, int ia, PairPa pa, Parray parray, int start, int end, float seed, /**/ Fo *fa, Farray farray) {
    enum {X, Y, Z};
    int ib;
    PairPa pb;
    Fo f;
    float rnd;

    for (ib = start; ib < end; ++ib) {
        if (ib >= ia) continue;

        fetch(parray, ib, &pb);

        if (!cutoff_range(pa, pb)) continue;

        rnd = rnd::mean0var1ii(seed, ia, ib);
        pair_force(params, pa, pb, rnd, /**/ &f);

        pair_add(&f, /**/ fa);

        farray_atomic_add<-1>(&f, ib, /**/ farray);
    }
}

template <typename Par, typename PArray, typename Farray, typename Fo>
_S_ void one_row(const Par *params, int3 L, int dz, int dy, int ia, int3 ca, PairPa pa, PArray parray, const int *start, float seed,
                               /**/ Fo *fa, Farray farray) {
    int3 cb;
    int enddx, startx, endx, cid0, bs, be;
    cb.z = ca.z + dz;
    cb.y = ca.y + dy;
    if (!valid_c(cb.z, L.z)) return;
    if (!valid_c(cb.y, L.y)) return;

    /* dx runs from -1 to enddx */
    enddx = (dz == 0 && dy == 0) ? 0 : 1;

    startx =     max(    0, ca.x - 1    );
    endx   = 1 + min(L.x-1, ca.x + enddx);

    cid0 = L.x * (cb.y + L.y * cb.z);

    bs = start[cid0 + startx];
    be = start[cid0 + endx];

    loop_pp(params, ia, pa, parray, bs, be, seed, /**/ fa, farray);
}

// unroll loop
template <typename Par, typename Parray, typename Farray>
__global__ void apply(Par params, int3 L, int n, Parray parray, const int *start, float seed, /**/ Farray farray) {
    enum {X, Y, Z};
    int ia;
    int3 ca;
    PairPa pa;
    auto fa = farray_fo0(farray);

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;

    fetch(parray, ia, &pa);
    ca = get_cid(L, &pa);

#define ONE_ROW(dz, dy) one_row (&params, L, dz, dy, ia, ca, pa, parray, start, seed, /**/ &fa, farray)

    ONE_ROW(-1, -1);
    ONE_ROW(-1,  0);
    ONE_ROW(-1,  1);
    ONE_ROW( 0, -1);
    ONE_ROW( 0,  0);

#undef ONE_ROW

    farray_atomic_add<1>(&fa, ia, farray);
}

#undef _S_
/* end src/fluforces/bulk/dev/main.h */
// #include "dev/map.h"
// #include "dev/xmain.h"
}

/* begin src/fluforces/bulk/imp/main.h */
void flocal_push_pp(const float4 *pp, BPaArray *a) {
    a->pp = pp;
    a->colors = false;
}

void flocal_push_cc(const int *cc, BPaArray *a) {
    a->cc = cc;
    a->colors = true;
}

static void tbarray_get_view(int n, BPaArray a, TBPaArray_v *v) {
    texo_setup(2*n, (float4*) a.pp, &v->pp);
}

static void tbarray_get_view(int n, BPaArray a, TBPaCArray_v *v) {
    texo_setup(2*n, (float4*) a.pp, &v->pp);
    texo_setup(n,      (int*) a.cc, &v->cc);
}

static void tbarray_fin_view(TBPaArray_v *v) {
    texo_destroy(&v->pp);
}

static void tbarray_fin_view(TBPaCArray_v *v) {
    texo_destroy(&v->pp);
    texo_destroy(&v->cc);
}

template <typename Par, typename Parray, typename Farray>
static void interactions(Par params, int3 L, int n, Parray parray, const int *start, RNDunif *rnd, /**/ Farray farray) {
    enum {THR=128};
    float seed;
    if (n <= 0) return;
    seed = rnd_get(rnd);

    KL(fbulk_dev::apply,
       (ceiln((n), THR), THR),
       (params, L, n, parray, start, seed, /**/ farray));
}

template <typename Par, typename Parray>
static void dispatch_stress(Par params, int3 L, int n, Parray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, n, parray, start, rnd, /**/ farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, n, parray, start, rnd, /**/ farray_v);
    }
}

static void apply_color(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    PairDPDC pv;
    TBPaCArray_v parray_v;

    pair_get_view_dpd_color(params, &pv);
    tbarray_get_view(n, parray, &parray_v);

    dispatch_stress(pv, L, n, parray_v, start, rnd, /**/ farray);

    tbarray_fin_view(&parray_v);
}

static void apply_grey(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    PairDPD pv;
    TBPaArray_v parray_v;
    pair_get_view_dpd(params, &pv);
    tbarray_get_view(n, parray, &parray_v);

    dispatch_stress(pv, L, n, parray_v, start, rnd, /**/ farray);

    tbarray_fin_view(&parray_v);
}

void flocal_apply(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd, /**/ const FoArray *farray) {
    if (parray.colors)
        apply_color(params, L, n, parray, start, rnd, /**/ farray);
    else
        apply_grey(params, L, n, parray, start, rnd, /**/ farray);
}
/* end src/fluforces/bulk/imp/main.h */
/* end src/fluforces/bulk/imp.cu */

#line 1 "src/fluforces/halo/imp.cu"

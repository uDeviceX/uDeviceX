// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/fluforces/halo/imp.cu */
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
/* begin src/d/api.h */
#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#ifndef UDX_FLU_HALO_DEVICE_API_COMMON
#define UDX_FLU_HALO_DEVICE_API_COMMON
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
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
#endif

/* begin src/frag/dev.h */
#ifndef UDX_DISTR_EXCH_FRAG_DEV_COMMON
#define UDX_DISTR_EXCH_FRAG_DEV_COMMON
#ifdef FRAG_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace frag_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace frag_dev {
  #define END }
#endif

BEGIN

enum {BAD_DIR = -2};

/* fragment id to direction                    */

_I_ int i2dx(int i) { return (i     + 2) % 3 - 1; }
_I_ int i2dy(int i) { return (i / 3 + 2) % 3 - 1; }
_I_ int i2dz(int i) { return (i / 9 + 2) % 3 - 1; }

_I_ void i2d3(int i, /**/ int d[3]) {
    enum {X, Y, Z};
    d[X] = i2dx(i);
    d[Y] = i2dy(i);
    d[Z] = i2dz(i);
}

/* direction to fragment id                    */
_I_ int d2i(int x, int y, int z) {
    return    ((x + 2) % 3)
        + 3 * ((y + 2) % 3)
        + 9 * ((z + 2) % 3);
}

_I_ int d32i(const int d[3])  {
    enum {X, Y, Z};
    return d2i(d[X], d[Y], d[Z]);
}

/* number of cells in direction x, y, z        */
_S_ int ncell0(int3 L, int x, int y, int z) {
    int nx, ny, nz;
    nx = x ? 1 : L.x;
    ny = y ? 1 : L.y;
    nz = z ? 1 : L.z;
    return nx * ny * nz;
}

/* number of cells in fragment i               */
_I_ int ncell(int3 L, int i) {
    int x, y, z;
    x = i2dx(i);
    y = i2dy(i);
    z = i2dz(i);
    return ncell0(L, x, y, z);
}

/* anti direction to fragment id                */
_I_ int antid2i(int x, int y, int z) {
    return d2i(-x, -y, -z);
}

/* anti fragment                                */
_I_ int anti(int i) {
    int x, y, z;
    x = i2dx(i);
    y = i2dy(i);
    z = i2dz(i);
    return antid2i(x, y, z);
}

/* [f]ragment [id] : where is `i' in sorted a[27]? */
_I_ int frag_get_fid(const int a[], const int i) {
    int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
/* end src/frag/dev.h */
#endif /* UDX_DISTR_EXCH_FRAG_DEV_COMMON */
/* begin src/frag/imp.h */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
enum {frag_bulk = 26};

struct int3;

namespace frag_hst {

// tag::i2d[]
int i2dx(int i);                 // <1>
int i2dy(int i);                 // <2>
int i2dz(int i);                 // <3>
void i2d3(int i, /**/ int d[3]); // <4>
// end::i2d[]

// tag::d2i[]
int d2i(int x, int y, int z); // <1>
int d32i(const int d[3]);     // <2>
// end::d2i[]

// tag::ncell[]
int ncell(int3 L, int i); // <1>
// end::ncell[]

// tag::anti[]
int antid2i(int x, int y, int z); // <1>
int anti(int i);                  // <2>
// end::anti[]

// tag::hst[]
void estimates(int3 L, int nfrags, float maxd, /**/ int *cap); // <1>
// end::hst[]

} // frag_hst
/* end src/frag/imp.h */
#endif /* UDX_DISTR_EXCH_FRAG_IMP_COMMON */

/* begin src/inc/def.h */
#ifndef UDX_DISTR_EXCH_DEF_COMMON
#define UDX_DISTR_EXCH_DEF_COMMON
#ifndef UDX_FLU_HALO_DEF_COMMON
#define UDX_FLU_HALO_DEF_COMMON
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
#endif /* UDX_DISTR_EXCH_DEF_COMMON */
#endif
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

/* begin src/inc/type.h */
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
#ifndef UDX_FLU_HALO_TYPE_COMMON
#define UDX_FLU_HALO_TYPE_COMMON
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
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
#endif
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

/* begin src/utils/kl.h */
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif /* UDX_DISTR_EXCH_KL_COMMON */

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
#ifndef UDX_FLU_PARRAY_COMMON
#define UDX_FLU_PARRAY_COMMON
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
#endif /* UDX_FLU_PARRAY_COMMON */
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

/* begin src/flu/type.h */
#ifndef UDX_FLU_TYPE_COMMON
#define UDX_FLU_TYPE_COMMON
/* helpers to communicate between halo interactions and exchanger */

namespace flu {

enum { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct LFrag { /* "local" fragment */
    PaArray parray;
    const int *ii; /* index */
    int n;
};

struct RFrag { /* "remote" fragment */
    PaArray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

struct RndFrag {
    float seed;
    int mask;
};

typedef Sarray<LFrag,   26> LFrag26;
typedef Sarray<RFrag,   26> RFrag26;
typedef Sarray<RndFrag, 26> RndFrag26;

} // flu
/* end src/flu/type.h */
#endif /* UDX_FLU_TYPE_COMMON */

/* begin src/fluforces/halo/imp.h */
struct PairParams;
struct int3;
struct FoArray;

void fhalo_apply(const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray);
/* end src/fluforces/halo/imp.h */

/* begin src/fluforces/halo/imp/type.h */
template<typename Parray>
struct LFrag_v { /* "local" fragment */
    Parray parray;
    const int *ii; /* index */
    int n;
};

template<typename Parray>
struct RFrag_v { /* "remote" fragment */
    Parray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

template <typename Parray>
using LFrag_v26 = Sarray<LFrag_v<Parray>, 26>;

template <typename Parray>
using RFrag_v26 = Sarray<RFrag_v<Parray>, 26>;
/* end src/fluforces/halo/imp/type.h */
namespace fhalo_dev {
/* begin src/fluforces/halo/dev/dbg.h */
template <typename Parray>
static __device__ void assert_frag(int3 L, int i, const RFrag_v<Parray> frag) {
    int xs, ys, zs; /* sizes */
    int dx, dy, dz;
    xs = frag.xcells; ys = frag.ycells; zs = frag.zcells;
    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    assert(xs * ys * zs == frag_dev::ncell(L, i));
    assert(frag_dev::d2i(dx, dy, dz) == i);
}

template <typename Parray>
static __device__ void assert_rc(int3 L, const RFrag_v<Parray> frag, int i, int row, int col, int jump) {
    /* i: base cell id */
    int fid, nmax;
    int dx, dy, dz;
    int j;
    assert(row == 1 || row == 2 || row == 3);
    assert(col == 1 || col == 2 || col == 3);

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    assert(dx == -1 || dx == 0 || dx == 1);
    assert(dy == -1 || dy == 0 || dy == 1);
    assert(dz == -1 || dz == 0 || dz == 1);
    fid  = frag_dev::d2i(dx, dy, dz);
    nmax = frag_dev::ncell(L, fid) + 1;

    for (j = 0 ; j < row; j++) {
        assert(i       < nmax);
        assert(i + col < nmax);
        i += jump;
    }
}
/* end src/fluforces/halo/dev/dbg.h */
/* begin src/fluforces/halo/dev/map.h */
struct Map { /* helps to find remote particle */
    int  org0, org1, org2;
    int cnt0, cnt1, cnt2;
};

/* true if `i' bigger than the number of remote particles */
static __device__ int endp(const Map m, int i) { return i >= m.cnt2; }
static __device__ int m2id(const Map m, int i) {
    /* return remote particle id */
    int m1, m2;
    int id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}

static __device__ int get(const int *a, int i) { return a[i]; }
static __device__ Map r2map0(const int *start, int i, int row, int col, int jump) {
    int j;
    int o[3] = {0}, c[3] = {0};
    Map m;
    assert(row == 1 || row == 2 || row == 3);
    assert(col == 1 || col == 2 || col == 3);

    for (j = 0 ; j < row; j++) {
        o[j] = get(start, i);
        c[j] = get(start, i + col) - o[j];
        i += jump;
    }
    c[1] += c[0]; c[2] += c[1];
    o[1] -= c[0]; o[2] -= c[1];

    m.org0 = o[0]; m.org1 = o[1]; m.org2 = o[2];
    m.cnt0 = c[0]; m.cnt1 = c[1]; m.cnt2 = c[2];
    return m;
}


static __device__ int facep(int dx, int dy, int dz) { /* face? */
    int x, y, z;
    x =  dx  && !dy && !dz;
    y =  !dx &&  dy && !dz;
    z =  !dx && !dy &&  dz;
    return x || y || z;
}
static __device__ void xyz2rc_face(int dx, int dy, int dz,
                                   int xc, int yc, int zc,
                                   int xs, int ys, int zs,
                                   /**/ int *prow, int *pcol, int *pjump) {
    int row, col, jump;
    if (!facep(dx, dy, dz)) {
        printf("xyz: %d %d %d\n", dx, dy, dz);
        assert(0);
    }
    if (dx) {
        row   = zs;
        col   = ys;
        jump = yc;
    } else if (dy) {
        row   = zs;
        col   = xs;
        jump = xc;
    } else { /* (dz) */
        row   = ys;
        col   = xs;
        jump = xc;
    }
    *prow = row; *pcol = col; *pjump = jump;
}

static __device__ void xyz2rc(int type,
                              int dx, int dy, int dz, /* fragment information */
                              int xc, int yc, int zc,
                              int xs, int ys, int zs, /* size */
                              /**/ int *prow, int *pcol, int *pjump) {
    int row, col, jump;
    if      (type == flu::FACE)
        xyz2rc_face(dx, dy, dz, xc, yc, zc, xs, ys, zs, /**/ &row, &col, &jump);
    else if (type == flu::EDGE) {
        row = jump = 1;
        col = max(xs, max(ys, zs));
    }
    else if (type == flu::CORNER) {
        row = col = jump = 1;
    } else {
        printf("%s:%d: illigal fragmant type: %d [%d %d %d]\n",
               __FILE__, __LINE__, type, dx, dy, dz);
        assert(0);
    }
    *prow = row; *pcol = col; *pjump = jump;
}

static __device__  void r2size(float r, int nc, int S, /**/ int *pl, int *ps) {
    int i, s, l;
    i = (int)(r + S/2);
    l = max(0,  i - 1);
    s = min(nc, i + 2) - l;
    *pl = l; *ps = s;
}

template <typename Parray>
static __device__ Map r2map(int3 L, const RFrag_v<Parray> frag, float x, float y, float z) {
    /* coordinate [r] to map */
    int id; /* base id */
    int xl, yl, zl; /* low */
    int xs, ys, zs; /* size */
    int dx, dy, dz;
    int xc, yc, zc;
    int row, col, jump;

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    xc = frag.xcells; yc = frag.ycells; zc = frag.zcells;

    id = 0; xs = 1; ys = 1; zs = 1;
    if (dz == 0) {r2size(z, zc, L.z, /**/ &zl, &zs); id += zl;}
    id *= yc;
    if (dy == 0) {r2size(y, yc, L.y, /**/ &yl, &ys); id += yl;}
    id *= xc;
    if (dx == 0) {r2size(x, xc, L.x, /**/ &xl, &xs); id += xl;}

    xyz2rc(frag.type,
           dx, dy, dz,
           xc, yc, zc,
           xs, ys, zs,
           &row, &col, &jump);
    assert_rc(L, frag, id, row, col, jump);
    return r2map0(frag.start, id, row, col, jump);
}
/* end src/fluforces/halo/dev/map.h */
/* begin src/fluforces/halo/dev/main.h */
template <typename Farray>
struct Fo {
    Farray farray;
    int i;
};

static __device__ float random(int aid, int bid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? aid : bid;
    a2 = mask ? bid : aid;
    return rnd::mean0var1uu(seed, a1, a2);
}

template<typename Par, typename Parray, typename PFo>
static __device__ void force0(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> bfrag, const Map m, const PairPa a, int aid,
                              /**/ PFo *fa) {
    PairPa b;
    PFo f;
    int i;
    int bid; /* id of b */
    float rndval;

    for (i = 0; !endp(m, i); i ++ ) {
        bid = m2id(m, i);
        parray_get(bfrag.parray, bid, /**/ &b);
        rndval = random(aid, bid, rnd.seed, rnd.mask);
        pair_force(&params, a, b, rndval, /**/ &f);
        pair_add(&f, /**/ fa);
    }
}

template<typename Par, typename Parray, typename Farray>
static __device__ void force1(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> frag, const Map m, const PairPa p, int id, Fo<Farray> f) {
    auto fa = farray_fo0(f.farray);
    force0(params, rnd, frag, m, p, id, /**/ &fa);
    farray_atomic_add<1>(&fa, f.i, f.farray);
}

template<typename Par, typename Parray, typename Farray>
static __device__ void force2(Par params, int3 L, const RFrag_v<Parray> frag, const flu::RndFrag rnd, PairPa p, int id, /**/ Fo<Farray> f) {
    Map m;
    m = r2map(L, frag, p.x, p.y, p.z);
    p.x -= frag.dx * L.x;
    p.y -= frag.dy * L.y;
    p.z -= frag.dz * L.z;

    force1(params, rnd, frag, m, p, id, /**/ f);
}

template <typename Farray>
static __device__ Fo<Farray> i2f(const int *ii, Farray farray, int i) {
    /* local id and index to force */
    Fo<Farray> f;
    f.i = ii[i];
    f.farray = farray;
    return f;
}

template <typename Parray, typename Farray>
static __device__ Fo<Farray> Lfrag2f(const LFrag_v<Parray> frag, Farray farray, int i) {
    return i2f(frag.ii, farray, i);
}

template <typename Par, typename Parray, typename Farray>
static __device__ void force3(Par params, int3 L, const LFrag_v<Parray> afrag, const RFrag_v<Parray> bfrag, const flu::RndFrag rnd, int i,
                              /**/ Farray farray) {
    PairPa p;
    Fo<Farray> f;
    parray_get(afrag.parray, i, &p);
    f = Lfrag2f(afrag, farray, i);
    force2(params, L, bfrag, rnd, p, i, f);
}

template <typename Par, typename Parray, typename Farray>
__global__ void apply(int fid, Par params, int3 L, const LFrag_v<Parray> lfrag, const RFrag_v<Parray> rfrag, const flu::RndFrag rnd,
                      /**/ Farray farray) {
    int i; /* particle id */

    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= lfrag.n) return;

    force3(params, L, lfrag, rfrag, rnd, i, /**/ farray);
}
/* end src/fluforces/halo/dev/main.h */
}
/* begin src/fluforces/halo/imp/main.h */
enum {
    PADDING = 16 /* warpsize / 2 */
};

template <typename Parray>
static void get_view_lfrag(const flu::LFrag26 f, LFrag_v26<Parray> *v) {
    Parray pav;
    LFrag_v<Parray> *vi;
    const flu::LFrag *fi;

    for (int i = 0; i < 26; ++i) {
        vi = &v->d[i];
        fi = &f.d[i];
        parray_get_view(&fi->parray, &pav);

        vi->parray = pav;
        vi->ii = fi->ii;
        vi->n  = fi->n;
    }
}

template <typename Parray>
static void get_view_rfrag(const flu::RFrag26 f, RFrag_v26<Parray> *v) {
    Parray pav;
    RFrag_v<Parray> *vi;
    const flu::RFrag *fi;

    for (int i = 0; i < 26; ++i) {
        vi = &v->d[i];
        fi = &f.d[i];
        parray_get_view(&fi->parray, &pav);

        vi->parray = pav;
        vi->start = fi->start;
        vi->dx = fi->dx;
        vi->dy = fi->dy;
        vi->dz = fi->dz;
        vi->xcells = fi->xcells;
        vi->ycells = fi->ycells;
        vi->zcells = fi->zcells;
        vi->type = fi->type;
    }
}

static bool is_colored(const flu::RFrag26 f) {
    return parray_is_colored(&(f.d[0].parray));
}

template <typename Par, typename Parray, typename Farray>
static void interactions(Par params, int3 L, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ Farray farray) {
    int n, i;
    for (i = 0; i < 26; ++i) {
        n = lfrags.d[i].n;
        KL(fhalo_dev::apply, (k_cnf(n)), (i, params, L, lfrags.d[i], rfrags.d[i], rrnd.d[i], /**/ farray));
    }
}

template <typename Par, typename Parray>
static void apply(Par params, int3 L, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {

    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, lfrags, rfrags, rrnd, /**/ farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, lfrags, rfrags, rrnd, /**/ farray_v);
    }
}

static void apply_grey(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    PairDPD pv;
    LFrag_v26<PaArray_v> lfragsv;
    RFrag_v26<PaArray_v> rfragsv;

    pair_get_view_dpd(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);

    apply(pv, L, lfragsv, rfragsv, rrnd, /**/ farray);
}

static void apply_color(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    PairDPDC pv;
    LFrag_v26<PaCArray_v> lfragsv;
    RFrag_v26<PaCArray_v> rfragsv;

    pair_get_view_dpd_color(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);

    apply(pv, L, lfragsv, rfragsv, rrnd, /**/ farray);
}

void fhalo_apply(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    if (is_colored(rfrags))
        apply_color(params, L, lfrags, rfrags, rrnd, /**/ farray);
    else
        apply_grey(params, L, lfrags, rfrags, rrnd, /**/ farray);
}
/* end src/fluforces/halo/imp/main.h */
/* end src/fluforces/halo/imp.cu */

#line 1 "src/fluforces/imp.cpp"

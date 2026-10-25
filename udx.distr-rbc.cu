// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/distr/rbc/imp.cu */
#include <mpi.h>
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


/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
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

/* begin src/rbc/type.h */
struct AreaVolume;
struct Particle;

// tag::quants[]
struct RbcQuants {
    int n, nc;               /* number of particles, cells                */
    int nt, nv;              /* number of triangles and vertices per mesh */
    int md;                  /* maximum valence of the mesh               */
    Particle *pp, *pp_hst;   /* vertices particles on host and device     */
    bool ids;                /* global ids active ?                       */
    int *ii;                 /* global ids on host                        */
    AreaVolume *area_volume; /* to compute area and volume */
};
// end::quants[]

/* end src/rbc/type.h */

/* begin src/frag/dev.h */
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
/* begin src/frag/imp.h */
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
/* begin src/comm/imp.h */
typedef unsigned char data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

/* Allocation mod for bags */
enum AllocMod {
    // tag::AllocMod[]
    HST_ONLY,   /* only host bags allocated                 */
    DEV_ONLY,   /* only device bags allocated               */
    PINNED,     /* both host and device pinned              */
    PINNED_HST, /* host pinned; no device memory            */
    PINNED_DEV, /* host pinned; device global memory on gpu */
    NONE        /* no allocation                            */
    // end::AllocMod[]
};

// tag::dBags[]
struct dBags {
    data_t *data[NBAGS]; /* data on the device         */
};
// end::dBags[]

// tag::hBags[]
struct hBags {
    data_t *data[NBAGS]; /* data on the host                    */
    int         *counts; /* size of the data                    */
    int capacity[NBAGS]; /* capacity of each frag (elem number) */
    size_t bsize;        /* size of one datum in bytes          */
};
// end::hBags[]

struct Comm;
struct CommBuffer;

// tag::alloc[]
void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db);
void comm_bags_fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db);
void comm_ini(MPI_Comm cart, /**/ Comm **c);
void comm_fin(/**/ Comm *c);
void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer**);
void comm_buffer_fin(CommBuffer*);
// end::alloc[]


void comm_buffer_set(int nbags, const hBags*, CommBuffer*);
void comm_buffer_get(const CommBuffer*, int nbags, hBags*);

// tag::communication[]
void comm_post_recv(hBags *b, Comm *c);           // <1>
void comm_post_send(const hBags *b, Comm *c);     // <2>

void comm_wait_recv(Comm *c, /**/ hBags *b);      // <3>
void comm_wait_send(Comm *c);                     // <4>
// end::communication[]

// tag::communication_b[]
void comm_post_recv(CommBuffer *cb, Comm *c);
void comm_post_send(const CommBuffer *cb, Comm *c);

void comm_wait_recv(Comm *c, /**/ CommBuffer *cb);
void comm_wait_send(Comm *c);
// end::communication_b[]

// tag::get[]
int    comm_get_number_capacity(int i, const hBags *b); // <1>
size_t comm_get_byte_capacity(int i, const hBags *b);   // <2>
// end::get[]

/* end src/comm/imp.h */
/* begin src/comm/utils.h */
// tag::convert[]
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
// end::convert[]
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
/* end src/comm/utils.h */

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
/* begin src/algo/minmax/imp.h */
struct Particle;
struct float3;
// tag::interface[]
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);
// end::interface[]
/* end src/algo/minmax/imp.h */

/* begin src/distr/map/type.h */
// tag::struct[]
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
// end::struct[]
/* end src/distr/map/type.h */
/* begin src/distr/map/dev.h */
// tag::int[]
/* exclusive scan */
template <int NCOUNTS>
static __global__ void dmap_scan(/**/ DMap m)              // <1>
// end::int[]
{
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0, cnt = 0;

    if (tid < NCOUNTS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * shfl_up(val, L) ;
    if (tid <= NCOUNTS) m.starts[tid] = val - cnt;
}

// tag::int[]
static __device__ int dmap_get_fid(int3 L, const float r[3])       // <2>
// end::int[]
{
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -L.x/2) + (r[X] >= L.x/2);
    y = -1 + (r[Y] >= -L.y/2) + (r[Y] >= L.y/2);
    z = -1 + (r[Z] >= -L.z/2) + (r[Z] >= L.z/2);
    return frag_dev::d2i(x, y, z);
}

// tag::int[]
static __device__ void dmap_add(int pid, int fid, DMap m)  // <3>
// end::int[]
{
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}
/* end src/distr/map/dev.h */
/* begin src/distr/map/imp.h */
// tag::int[]
void dmap_ini(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin(int nfrags, /**/ DMap *m);
void dmap_reini(int nfrags, /**/ DMap m);
void dmap_download_counts(int nfrags, /**/ DMap *m);
void dmap_ini_host(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin_host(int nfrags, /**/ DMap *m);
void dmap_reini_host(int nfrags, /**/ DMap m);
void dmap_D2H(int nfrags, const DMap *d, /**/ DMap *h);
// end::int[]
/* end src/distr/map/imp.h */
/* begin src/distr/common/imp.h */
struct Particle;
struct int3;

// tag::int[]
void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf); // <1>
void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp); // <3>
// end::int[]

/* end src/distr/common/imp.h */

/* begin src/distr/rbc/imp.h */
struct DRbcPack;
struct DRbcComm;
struct DRbcUnpack;

// tag::interface[]
void drbc_pack_ini(bool ids, int3 L, int maxc, int nv, DRbcPack **p);
void drbc_comm_ini(MPI_Comm comm, /**/ DRbcComm **c);
void drbc_unpack_ini(bool ids, int3 L, int maxc, int nv, DRbcUnpack **u);

void drbc_pack_fin(DRbcPack *p);
void drbc_comm_fin(DRbcComm *c);
void drbc_unpack_fin(DRbcUnpack *u);

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p);
void drbc_pack(const RbcQuants *q, /**/ DRbcPack *p);
void drbc_download(DRbcPack *p);

void drbc_post_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_post_send(DRbcPack *p, DRbcComm *c);
void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_wait_send(DRbcComm *c);

void drbc_unpack_bulk(const DRbcPack *p, /**/ RbcQuants *q);
void drbc_unpack_halo(const DRbcUnpack *u, /**/ RbcQuants *q);
// end::interface[]
/* end src/distr/rbc/imp.h */

/* begin src/distr/rbc/dev.h */
namespace drbc_dev {

__global__ void build_map(int3 L, int n, const float3 *minext, const float3 *maxext, /**/ DMap m) {
    enum {X, Y, Z};
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float r[3];
    float3 mine, maxe;

    mine = minext[i];
    maxe = maxext[i];

    r[X] = 0.5f * (mine.x + maxe.x);
    r[Y] = 0.5f * (mine.y + maxe.y);
    r[Z] = 0.5f * (mine.z + maxe.z);

    fid = dmap_get_fid(L, r);
    dmap_add(i, fid, /**/ m);
}

}
/* end src/distr/rbc/dev.h */
/* begin src/distr/rbc/imp/type.h */
// tag::struct[]
enum {
    MAX_NHBAGS = 2,
    MAX_NDBAGS = 1
};

struct DRbcPack {
    DMap map;
    float3 *minext, *maxext;
    dBags dbags[MAX_NDBAGS], *dpp;
    hBags hbags[MAX_NHBAGS], *hpp, *hii;
    int nbags;
    CommBuffer *hbuf;
    DMap hmap; /* host map for ids */

    int3 L; /* subdomain size */
};

struct DRbcComm {
    Comm *comm;
};

struct DRbcUnpack {
    hBags hbags[MAX_NHBAGS], *hpp, *hii;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};
// end::struct[]
/* end src/distr/rbc/imp/type.h */
/* begin src/distr/rbc/imp/ini.h */
static void get_num_capacity(int maxnc, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxnc;
}

void drbc_pack_ini(bool ids, int3 L, int maxnc, int nv, DRbcPack **pack) {
    int i, numc[NBAGS];
    DRbcPack *p;
    EMALLOC(1, pack);
    p = *pack;

    p->L= L;

    get_num_capacity(maxnc, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    i = 0;
    p->hpp = &p->hbags[i];
    p->dpp = &p->dbags[i++];
    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ p->hpp, p->dpp));

    Dalloc(&p->minext, maxnc);
    Dalloc(&p->maxext, maxnc);

    p->hii = NULL;
    if (ids) {
        p->hii = &p->hbags[i++];
        UC(dmap_ini_host(NBAGS, numc, /**/ &p->hmap));
        UC(comm_bags_ini(HST_ONLY, HST_ONLY, sizeof(int), numc, /**/ p->hii, NULL));
    }
    p->nbags = i;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void drbc_comm_ini(MPI_Comm cart, /**/ DRbcComm **com) {
    DRbcComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void drbc_unpack_ini(bool ids, int3 L, int maxnc, int nv, DRbcUnpack **unpack) {
    int i, numc[NBAGS];
    DRbcUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;

    get_num_capacity(maxnc, /**/ numc);

    i = 0;
    u->hpp = &u->hbags[i++];
    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ u->hpp, NULL));

    u->hii = NULL;
    if (ids) {
        u->hii = &u->hbags[i++];
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), numc, /**/ u->hii, NULL));
    }
    u->nbags = i;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}
/* end src/distr/rbc/imp/ini.h */
/* begin src/distr/rbc/imp/fin.h */
void drbc_pack_fin(DRbcPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ p->hpp, p->dpp));
    Dfree(p->minext);
    Dfree(p->maxext);

    if (p->hii) {
        UC(dmap_fin_host(NBAGS, /**/ &p->hmap));
        UC(comm_bags_fin(HST_ONLY, HST_ONLY, /**/ p->hii, NULL));
    }
    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void drbc_comm_fin(DRbcComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void drbc_unpack_fin(DRbcUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, /**/ u->hpp, NULL));
    if (u->hii)
        UC(comm_bags_fin(HST_ONLY, NONE, /**/ u->hii, NULL));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}
/* end src/distr/rbc/imp/fin.h */
/* begin src/distr/rbc/imp/map.h */
static void build_map(int3 L, int n, const float3 *minext, const float3 *maxext, /**/ DMap m) {
    UC(dmap_reini(NBAGS, /**/ m));
    KL(drbc_dev::build_map, (k_cnf(n)), (L, n, minext, maxext, /**/ m));
    KL(dmap_scan<NBAGS>, (1, 32), (/**/ m));
}

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p) {
    minmax(pp, nv, nc, /**/ p->minext, p->maxext);
    build_map(p->L, nc, p->minext, p->maxext, /**/  p->map);
}
/* end src/distr/rbc/imp/map.h */
/* begin src/distr/rbc/imp/pack.h */
static void pack_pp(const DMap m, int nc, int nv, const Particle *pp, /**/ dBags *bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(*bags, &wrap);
    UC(dcommon_pack_pp_packets(nc, nv, pp, m, /**/ wrap));
}

/* all data (including map) on host */
static void pack_ii(const DMap m, int, const int *ii, /**/ hBags *bags) {
    Sarray<int*, NBAGS> wrap;
    bag2Sarray(*bags, &wrap);

    int i, c, j, src;
    for (i = 0; i < NBAGS; ++i) {
        c = m.counts[i];
        for (j = 0; j < c; ++j) {
            src = m.ids[i][j];
            wrap.d[i][j] = ii[src];
        }
    }
}

void drbc_pack(const RbcQuants *q, /**/ DRbcPack *p) {
    UC(pack_pp(p->map, q->nc, q->nv, q->pp, /**/ p->dpp));
    if (p->hii) {
        UC(dmap_D2H(NBAGS, &p->map, /**/ &p->hmap));
        dSync();
        UC(pack_ii(p->hmap, q->nc, q->ii, /**/ p->hii));
    }
}

void drbc_download(DRbcPack *p) {
    size_t sz = NBAGS * sizeof(int);
    CC(d::Memcpy(p->hpp->counts, p->map.counts, sz, D2H));
    if (p->hii)
        memcpy(p->hii->counts, p->hmap.counts, sz);
}
/* end src/distr/rbc/imp/pack.h */
/* begin src/distr/rbc/imp/com.h */
void drbc_post_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void drbc_post_send(DRbcPack *p, DRbcComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void drbc_wait_send(DRbcComm *c) {
    UC(comm_wait_send(c->comm));
}
/* end src/distr/rbc/imp/com.h */
/* begin src/distr/rbc/imp/unpack.h */
static int unpack_bulk_pp(const DRbcPack *p, /**/ Particle *pp) {
    int nc = p->hpp->counts[frag_bulk];
    void *src = p->dpp->data[frag_bulk];
    size_t sz = nc * p->hpp->bsize;

    if (nc)
        CC(d::MemcpyAsync(pp, src, sz, D2D));
    return nc;
}

static void unpack_bulk_ii(const DRbcPack *p, /**/ int *ii) {
    int nc = p->hii->counts[frag_bulk];
    void *src = p->hii->data[frag_bulk];
    memcpy(ii, src, nc * sizeof(int));
}

void drbc_unpack_bulk(const DRbcPack *p, /**/ RbcQuants *q) {
    int nc, nv, n;
    nv = q->nv;

    nc = unpack_bulk_pp(p, /**/ q->pp);
    n = nc * nv;

    if (p->hii)
        unpack_bulk_ii(p, /**/ q->ii);

    q->nc = nc;
    q->n = n;
}

static int unpack_halo_pp(int3 L, int nc0, int nv, const hBags *hpp, /**/ Particle *pp) {
    void *src;
    Particle *dst;
    int i, s, c, n;
    size_t sz, bsz;
    bsz = hpp->bsize;

    s = nc0;
    for (i = 0; i < NFRAGS; ++i) {
        c   = hpp->counts[i];
        src = hpp->data[i];
        dst = pp + s * nv;

        if (c) {
            n = c * nv;
            sz = c * bsz;
            CC(d::MemcpyAsync(dst, src, sz, H2D));
            dcommon_shift_one_frag(L, n, i, /**/ dst);
        }
        s += c;
    }
    return s;
}

static void unpack_halo_ii(int nc0, const hBags *hii, /**/ int *ii) {
    void *src;
    int i, s, c;

    s = nc0;
    for (i = 0; i < NFRAGS; ++i) {
        c   = hii->counts[i];
        src = hii->data[i];
        memcpy(ii + s, src, c * sizeof(int));
        s += c;
    }
}

void drbc_unpack_halo(const DRbcUnpack *u, /**/ RbcQuants *q) {
    int nc0, nc, nv;

    nc0 = q->nc;
    nv  = q->nv;

    nc = unpack_halo_pp(u->L, nc0, nv, u->hpp, /**/ q->pp);

    if (u->hii)
        unpack_halo_ii(nc0, u->hii, /**/ q->ii);

    q->n = nc * nv;
    q->nc = nc;
}
/* end src/distr/rbc/imp/unpack.h */
/* end src/distr/rbc/imp.cu */

#line 1 "src/distr/rig/imp.cu"

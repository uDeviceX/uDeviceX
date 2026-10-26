
// Grouped editable compilation unit for wall exchange, forces, drivers, and SDF.


// Grouped editable compilation unit for wall exchange, force, and driver code.


// Grouped editable compilation unit for wall exchange and force handling.


#include "udx.config.cuh"

#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include <vector_types.h>

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

#ifndef UDX_WALL_DEF_COMMON
#define UDX_WALL_DEF_COMMON
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
#endif

enum {frag_bulk = 26};

struct int3;

namespace frag_hst {

int i2dx(int i);                 // <1>
int i2dy(int i);                 // <2>
int i2dz(int i);                 // <3>
void i2d3(int i, /**/ int d[3]); // <4>

int d2i(int x, int y, int z); // <1>
int d32i(const int d[3]);     // <2>

int ncell(int3 L, int i); // <1>

int antid2i(int x, int y, int z); // <1>
int anti(int i);                  // <2>

void estimates(int3 L, int nfrags, float maxd, /**/ int *cap); // <1>

} // frag_hst
typedef unsigned char data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

/* Allocation mod for bags */
enum AllocMod {
    HST_ONLY,   /* only host bags allocated                 */
    DEV_ONLY,   /* only device bags allocated               */
    PINNED,     /* both host and device pinned              */
    PINNED_HST, /* host pinned; no device memory            */
    PINNED_DEV, /* host pinned; device global memory on gpu */
    NONE        /* no allocation                            */
};

struct dBags {
    data_t *data[NBAGS]; /* data on the device         */
};

struct hBags {
    data_t *data[NBAGS]; /* data on the host                    */
    int         *counts; /* size of the data                    */
    int capacity[NBAGS]; /* capacity of each frag (elem number) */
    size_t bsize;        /* size of one datum in bytes          */
};

struct Comm;
struct CommBuffer;

void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db);
void comm_bags_fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db);
void comm_ini(MPI_Comm cart, /**/ Comm **c);
void comm_fin(/**/ Comm *c);
void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer**);
void comm_buffer_fin(CommBuffer*);


void comm_buffer_set(int nbags, const hBags*, CommBuffer*);
void comm_buffer_get(const CommBuffer*, int nbags, hBags*);

void comm_post_recv(hBags *b, Comm *c);           // <1>
void comm_post_send(const hBags *b, Comm *c);     // <2>

void comm_wait_recv(Comm *c, /**/ hBags *b);      // <3>
void comm_wait_send(Comm *c);                     // <4>

void comm_post_recv(CommBuffer *cb, Comm *c);
void comm_post_send(const CommBuffer *cb, Comm *c);

void comm_wait_recv(Comm *c, /**/ CommBuffer *cb);
void comm_wait_send(Comm *c);

int    comm_get_number_capacity(int i, const hBags *b); // <1>
size_t comm_get_byte_capacity(int i, const hBags *b);   // <2>


#ifndef UDX_WALL_TYPE_COMMON
#define UDX_WALL_TYPE_COMMON
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
#endif

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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

static void exch_estimate(int numdensity, int3 L, int cap[]) {
    enum {SAFETY_FACTOR = 2};
    int i, nc, c;
    for (i = 0; i < NFRAGS; ++i) {
        nc = frag_hst::ncell(L, i);
        nc *= frag_hst::i2dx(i) ? XWM : 1;
        nc *= frag_hst::i2dy(i) ? YWM : 1;
        nc *= frag_hst::i2dz(i) ? ZWM : 1;
        c = numdensity * nc;
        cap[i] = SAFETY_FACTOR * c;
    }
}

static void shift(int3 L, int fid, float r[3]) {
    enum {X, Y, Z};
    int d[3];
    frag_hst::i2d3(fid, d);
    r[X] += d[X] * L.x;
    r[Y] += d[Y] * L.y;
    r[Z] += d[Z] * L.z;
}

static bool is_inside(int3 LW, const Particle p) {
    enum {X, Y, Z};
    return
        p.r[X] >= -0.5 * LW.x && p.r[X] < 0.5 * LW.x &&
        p.r[Y] >= -0.5 * LW.y && p.r[Y] < 0.5 * LW.y &&
        p.r[Z] >= -0.5 * LW.z && p.r[Z] < 0.5 * LW.z;
}

static void fill_bags(int3 L, int n, const Particle *pp, hBags *b) {
    int i, j, *cc, c;
    Particle p0, p, **dst;
    int3 LW;
    LW.x = L.x + 2*XWM;
    LW.y = L.y + 2*YWM;
    LW.z = L.z + 2*ZWM;

    cc  = b->counts;
    dst = (Particle **) b->data;

    memset(cc, 0, NBAGS * sizeof(int));

    for (i = 0; i < n; ++i) {
        p0 = pp[i];
        for (j = 0; j < NFRAGS; ++j) {
            p = p0;
            shift(L, j, p.r);
            if (is_inside(LW, p)) {
                c = cc[j] ++;
                dst[j][c] = p;
            }
        }
    }
}

static void communicate(const hBags *s, Comm *c, hBags *r) {
    UC(comm_post_send(s, c));
    UC(comm_post_recv(r, c));
    UC(comm_wait_send(c));
    UC(comm_wait_recv(c, /**/ r));
}

static void check_counts(int maxn, int n0, const hBags *b) {
    int i, c = n0;
    for (i = 0; i < NFRAGS; ++i) c += b->counts[i];
    if (c > maxn)
        ERR("Too many particles for wall : %d / %d\n", c, maxn);
}

static void unpack(const hBags *b, /*io*/ int *n, Particle *pp) {
    int i, j, k, c;
    const Particle *src;
    k = *n;
    for (j = 0; j < NFRAGS; ++j) {
        c = b->counts[j];
        src = (const Particle *) b->data[j];
        for (i = 0; i < c; ++i) {
            pp[k] = src[i];
            ++k;
        }
    }
    *n = k;
}

/* exchange pp(hst) between processors to get a wall margin */
void wall_exch_pp(MPI_Comm cart, int numdensity, int3 L, int maxn, /*io*/ Particle *pp, int *n) {
    hBags send, recv;
    Comm *com;
    int capacity[NBAGS];

    exch_estimate(numdensity, L, capacity);

    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, &send, NULL));
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, &recv, NULL));
    UC(comm_ini(cart, &com));

    fill_bags(L, *n, pp, /**/ &send);
    communicate(&send, /**/ com, &recv);
    check_counts(maxn, *n, &recv);
    unpack(&recv, /**/ n, pp);

    UC(comm_bags_fin(HST_ONLY, NONE, &send, NULL));
    UC(comm_bags_fin(HST_ONLY, NONE, &recv, NULL));
    UC(comm_fin(com));
}





#include <stdio.h>
#include <assert.h>

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


#ifndef UDX_WALL_DEF_COMMON
#define UDX_WALL_DEF_COMMON
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
#endif
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
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

#ifndef UDX_WALL_DEVICE_API_COMMON
#define UDX_WALL_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif
/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
/* B.1. Function Type Qualifiers */
/* B.2. Variable Type Qualifiers */
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif

/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}



/* TODO: header includes header */

#ifndef UDX_WALL_TYPE_COMMON
#define UDX_WALL_TYPE_COMMON
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
#endif
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
#ifndef UDX_WALL_TEXO_COMMON
#define UDX_WALL_TEXO_COMMON
/* 1D texture object template */
static void sz_check(int n) {
    if (n <            0 ) ERR("too small texo: %d", n);
    if (n > MAX_TEXO_SIZE) ERR("too big texo: %d > %d", n, MAX_TEXO_SIZE);
}

template<typename T>
struct Texo {
    cudaTextureObject_t d;
};

template<typename T>
void texo_setup(int n, T *data, /**/ Texo<T> *to)
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
void texo_destroy(/**/ Texo<T> *to)
{
    CC(cudaDestroyTextureObject(to->d));
}
template<typename T>
__device__ __forceinline__
const T texo_fetch(const Texo<T> t, const int i)
{
    return Tfetch(T, t.d, i);
}
#endif

struct RNDunif;                                             // <1>
void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r); // <2>
void rnd_fin(RNDunif *r);                                   // <3>
float rnd_get(RNDunif *r);                                  // <4>
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
#ifndef UDX_WALL_MATH_COMMON
#define UDX_WALL_MATH_COMMON
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
#endif /* UDX_WALL_MATH_COMMON */

#ifndef UDX_WALL_COORDS_TYPE_COMMON
#define UDX_WALL_COORDS_TYPE_COMMON
struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
#endif
#ifndef UDX_WALL_COORDS_DEV_COMMON
#define UDX_WALL_COORDS_DEV_COMMON
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

#endif /* UDX_WALL_COORDS_DEV_COMMON */
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);

/* edges of the sub-domain in global coordinates */

int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);

/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);

#ifndef UDX_WALL_WVEL_TYPE_COMMON
#define UDX_WALL_WVEL_TYPE_COMMON
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

#endif
struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

#ifndef UDX_WALL_WVEL_ENUM
#define UDX_WALL_WVEL_ENUM
enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};
#endif

void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);

void wvel_step_ini(WvelStep **wv);
void wvel_step_fin(WvelStep *wv);

void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
void wvel_set_timestep(float dt, Wvel *vw);

void wvel_set_conf(const Config *cfg, Wvel *vw);

void wvel_get_step(float t, const Wvel *wv, /**/ WvelStep *);

int wvel_get_type(const WvelStep *w);

void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v);
#ifndef UDX_WALL_WVEL_DEV_COMMON
#define UDX_WALL_WVEL_DEV_COMMON
static __device__ void wvel(WvelCste_v p, Coords_v c, float3 r, /**/ float3 *v)
{
    *v = p.u;
}

static __device__ void wvel(WvelShear_v p, Coords_v c, float3 r, /**/ float3 *v)
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

static __device__ void wvel(WvelHS_v p, Coords_v c, float3 r, /**/ float3 *v)
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

template <typename Wvel_v>
static __device__ void bounce_vel(Wvel_v wv, Coords_v c, float3 rw, /* io */ float3* v)
{
    float3 vw;
    wvel(wv, c, rw, /**/ &vw);
    v->x = 2 * vw.x - v->x;
    v->y = 2 * vw.y - v->y;
    v->z = 2 * vw.z - v->z;
}
#endif /* UDX_WALL_WVEL_DEV_COMMON */

#ifndef UDX_WALL_TFORM_TYPE_COMMON
#define UDX_WALL_TFORM_TYPE_COMMON
struct Tform_v { float o[3], s[3]; };
#endif
#ifndef UDX_WALL_TFORM_DEV_COMMON
#define UDX_WALL_TFORM_DEV_COMMON
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
#endif /* UDX_WALL_TFORM_DEV_COMMON */

#ifndef UDX_WALL_TEX3D_TYPE_COMMON
#define UDX_WALL_TEX3D_TYPE_COMMON
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
#endif
#ifndef UDX_WALL_SDF_TYPE_COMMON
#define UDX_WALL_SDF_TYPE_COMMON
/* view */
struct Sdf_v {
    Tex3d_v tex;
    Tform_v t;
    float cheap_threshold;
};
#endif
#ifndef UDX_WALL_SDF_DEV_COMMON
#define UDX_WALL_SDF_DEV_COMMON
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

_I_ float3 sdf_ugrad(const Sdf_v *texsdf, const float3 *r) // <1>
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

_I_ bool sdf_far(const Sdf_v *sdf, float x, float y, float z) // <2>
{
    return cheap_sdf(sdf, x, y, z) <= sdf->cheap_threshold - 1 ;
}

_I_ float sdf_eval(const Sdf_v *sdf0, float x, float y, float z)  // <3>
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
#endif /* UDX_WALL_SDF_DEV_COMMON */

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

template <typename Param, typename Fo>
_I_ void pair_force(const Param *p, PairPa a, PairPa b, float rnd, /**/ Fo *f)
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


_I_ void pair_add(const PairFo *b, /**/ PairFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;
}

_I_ void pair_add(const PairSFo *b, /**/ PairSFo *a)
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
struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p); // <1>
void pair_set_lj(float sigma, float eps, PairParams *p); // <2>
void pair_set_adhesion(float k1, float k2, PairParams *p);

void pair_set_conf(const Config *, const char *name_space, PairParams *);

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);

void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);

struct PaArray_v {
    const float *pp;
};

struct PaCArray_v {
    const float *pp;
    const int *cc;   /* colors */
};
struct PaArray {
    bool colors;
    const float *pp;
    const int *cc;
};

struct PaArray_v;
struct PaCArray_v;

struct Particle;

void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);

bool parray_is_colored(const PaArray *a);

void parray_get_view(const PaArray *a, PaArray_v *v);  // <1>
void parray_get_view(const PaArray *a, PaCArray_v *v); // <2>
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

_I_ void parray_get(PaArray_v a, int i, /**/ PairPa *p) // <1>
{
    parray_get_pp(a, i, /**/ p);
}

_I_ void parray_get(PaCArray_v a, int i, /**/ PairPa *p) // <2>
{
    parray_get_pp(a, i, /**/ p);
    p->color = a.cc[i];
}

#undef _S_
#undef _I_

struct FoArray_v {
    float *ff;
};

struct FoSArray_v {
    float *ff;
    float *ss;
};
struct FoArray {
    bool stress;
    float *ff;
    float *ss; /* stress */
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);

void farray_clear(FoArray *);

bool farray_has_stress(const FoArray *a);

void farray_get_view(const FoArray *a, FoArray_v *v);  // <1>
void farray_get_view(const FoArray *a, FoSArray_v *v); // <2>
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

template <int S>
_I_ void farray_add(const PairFo *f, int i, /**/ FoArray_v a) // <1>
{
    farray_add_force<S>(f, i, /**/ a);
}

template <int S>
_I_ void farray_add(const PairSFo *f, int i, /**/ FoSArray_v a) // <2>
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

template <int S>
_I_ void farray_atomic_add(const PairFo *f, int i, /**/ FoArray_v a)
{
    farray_atomic_add_force<S>(f, i, /**/ a);
}

template <int S>
_I_ void farray_atomic_add(const PairSFo *f, int i, /**/ FoSArray_v a)
{
    farray_atomic_add_force<S>(f, i, /**/ a);
    farray_atomic_add_stress(f, i, /**/ a);
}


_I_ PairFo farray_fo0(FoArray_v)
{
    PairFo f;
    f.x = f.y = f.z = 0;
    return f;
}

_I_ PairSFo farray_fo0(FoSArray_v)
{
    PairSFo f;
    f.x = f.y = f.z = 0;
    f.sxx = f.sxy = f.sxz = 0;
    f.syy = f.syz = f.szz = 0;
    return f;
}

#undef _I_
#undef _S_

#ifndef UDX_WALL_KL_COMMON
#define UDX_WALL_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif
struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct RNDunif;

/* shared wall-force view types */
#ifndef UDX_WALL_FORCE_TYPE_COMMON
#define UDX_WALL_FORCE_TYPE_COMMON
struct WallForce { /* local wall data */
    Sdf_v sdf_v;
    Texo<int> start;
    Texo<float4> pp;
    int n;
    int3 L;
};

struct WallRepulsePrm {
    float l;
};

void wall_force_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                      /**/ const FoArray*); // <1>

void wall_force_adhesion_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                               /**/ const FoArray*);

void wall_force_repulse(Sdf_v, WallRepulsePrm, long n, const PaArray*, const FoArray*); // <2>
#endif

struct WallMap { /* helps to find remote particle */
    int org0, org1, org2;
    int str0, str1, str2;
};
static __device__ int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }

static __device__ void ini(int3 L, int zplane, const Texo<int> texstart, int w_n, float x, float y, float z, /**/ WallMap *m) {
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
/* true if `i' is bigger than the number of remote particles */
static __device__ int endp(const WallMap m, int i) { return i >= m.str2; }
static __device__ int m2id(const WallMap m, int i) {
    /* return remote particle id */
    int m1, m2, id;
    m1 = (i >= m.str0);
    m2 = (i >= m.str1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}
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
    WallMap m;
    PairPa b;  /* wall particles */
    Fo f;
    float rnd;
    int i, bid;

    if (sdf_far(&wa.sdf_v, a.x, a.y, a.z)) return;

    ini(wa.L, zplane, wa.start, wa.n, a.x, a.y, a.z, /**/ &m);

    for (i = 0; !endp(m, i); ++i) {
        bid = m2id(m, i);
        fetch_wall(wv, c, wa.pp, bid, /**/ &b);
        rnd = rnd::mean0var1ii(seed, aid, bid);
        pair_force(&params, a, b, rnd, /**/ &f);
        pair_add(&f, fa);
    }
}

template <typename Par, typename Wvel_v, typename Parray, typename Farray>
__global__ void wall_force_kernel(Par params, Wvel_v wv, Coords_v c, Parray parray, int np, float seed, WallForce wa, /**/ Farray farray) {
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

__global__ void wall_repulse_kernel(WallRepulsePrm wr, int n, const Particle *pp, Sdf_v sdf_v, /**/ Force *ff) {
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

enum {
    TPP = 3 /* # threads per particle */
};

template <typename Par, typename Wvel_v, typename Parray, typename Farray>
static void interactions(Par params, Wvel_v wv, const Coords *c, Parray parray, int n, RNDunif *rnd, WallForce wa, /**/ Farray farray) {
    Coords_v coordsv;
    coords_get_view(c, &coordsv);

    KL(wall_force_kernel,
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
    KL(wall_repulse_kernel,
       (k_cnf(n)),
       (wr, n, (const Particle*) pa->pp, sdf_v, (Force*) fo->ff));
}






#include <mpi.h>
#include <stdio.h>
#include <assert.h>

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

#ifndef UDX_WALL_DEF_COMMON
#define UDX_WALL_DEF_COMMON
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
#endif
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

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
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
#ifndef UDX_WALL_DEVICE_API_COMMON
#define UDX_WALL_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif
/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
/* B.1. Function Type Qualifiers */
/* B.2. Variable Type Qualifiers */
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif

/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}



/* TODO: header includes header */

#ifndef UDX_WALL_TYPE_COMMON
#define UDX_WALL_TYPE_COMMON
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
#endif
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
#ifndef UDX_WALL_KL_COMMON
#define UDX_WALL_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif
#ifndef UDX_WALL_TEXO_COMMON
#define UDX_WALL_TEXO_COMMON
/* 1D texture object template */
static void sz_check(int n) {
    if (n <            0 ) ERR("too small texo: %d", n);
    if (n > MAX_TEXO_SIZE) ERR("too big texo: %d > %d", n, MAX_TEXO_SIZE);
}

template<typename T>
struct Texo {
    cudaTextureObject_t d;
};

template<typename T>
void texo_setup(int n, T *data, /**/ Texo<T> *to)
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
void texo_destroy(/**/ Texo<T> *to)
{
    CC(cudaDestroyTextureObject(to->d));
}
#endif

struct Config;
struct float3;
struct int3;

void conf_ini(/**/ Config **c);
void conf_fin(/**/ Config *c);

void conf_read(int argc, char **argv, /**/ Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);     // <1>
void conf_write_history(const Config *cfg, FILE *stream); // <2>

struct RNDunif;                                             // <1>
void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r); // <2>
void rnd_fin(RNDunif *r);                                   // <3>
float rnd_get(RNDunif *r);                                  // <4>

#ifndef UDX_WALL_COORDS_TYPE_COMMON
#define UDX_WALL_COORDS_TYPE_COMMON
struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
#endif
#ifndef UDX_WALL_WVEL_TYPE_COMMON
#define UDX_WALL_WVEL_TYPE_COMMON
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

#endif

#ifndef UDX_WALL_TFORM_TYPE_COMMON
#define UDX_WALL_TFORM_TYPE_COMMON
struct Tform_v { float o[3], s[3]; };
#endif
#ifndef UDX_WALL_TEX3D_TYPE_COMMON
#define UDX_WALL_TEX3D_TYPE_COMMON
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
#endif
#ifndef UDX_WALL_SDF_TYPE_COMMON
#define UDX_WALL_SDF_TYPE_COMMON
/* view */
struct Sdf_v {
    Tex3d_v tex;
    Tform_v t;
    float cheap_threshold;
};
#endif
struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);

void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*); // <1>
void sdf_get_view(const Sdf*, /**/ Sdf_v*);                  // <2>

void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp); // <1>
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);          // <2>
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);                          // <3>

void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);

enum {
    ALIVE = 0,
    DEAD  = 1
};

/* structure of particle                           */
/* optional "deathlist" for particles to be killed */

struct PartList {
    const Particle *pp;   /* particles array          */
    const int *deathlist; /* what pp is dead or alive */
};
struct Scan;

void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
struct Clist;
struct ClistMap;

void clist_ini(int LX, int LY, int LZ, /**/ Clist **c);
void clist_fin(/**/ Clist *c);

void clist_ini_map(int maxp, int nA, const Clist *c, /**/ ClistMap **m);
void clist_fin_map(ClistMap *m);

const int* clists_get_ss(const Clist *c);
const int* clists_get_cc(const Clist *c);
int3       clists_get_dim(const Clist *c);
int        clists_get_n(const Clist *c);
const uint* clist_get_ids(const ClistMap *m);

void clist_ini_counts(Clist *c);
void clist_subindex(bool project, int aid, int n, const PartList lp, /**/ Clist *c, ClistMap *m);
void clist_build_map(const int nn[], /**/ Clist *c, ClistMap *m);

/* special for fluid distribution */
void clist_subindex_local(int n, const PartList lp, /**/ Clist *c, ClistMap *m);
void clist_subindex_remote(int n, const PartList lp, /**/ Clist *c, ClistMap *m);

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout, /**/ Particle *ppout);
void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout, /**/ int *iiout);

/* quick cell build for single array */
void clist_build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, ClistMap *m);

struct Coords;
struct Particle;
struct Rigid;

enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};

typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);

void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp); // <1>
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);      // <2>
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);    // <3>
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter); // <4>

void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);

struct int3;
struct Particle;
void wall_exch_pp(MPI_Comm cart, int numdensity, int3 L, int maxn, /*io*/ Particle *pp, int *n);
struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct RNDunif;

/* shared wall-force view types */
#ifndef UDX_WALL_FORCE_TYPE_COMMON
#define UDX_WALL_FORCE_TYPE_COMMON
struct WallForce { /* local wall data */
    Sdf_v sdf_v;
    Texo<int> start;
    Texo<float4> pp;
    int n;
    int3 L;
};

struct WallRepulsePrm {
    float l;
};

void wall_force_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                      /**/ const FoArray*); // <1>

void wall_force_adhesion_apply(const PairParams*, const WvelStep*, const Coords*, const PaArray*, int n, RNDunif*, WallForce,
                               /**/ const FoArray*);

void wall_force_repulse(Sdf_v, WallRepulsePrm, long n, const PaArray*, const FoArray*); // <2>
#endif

struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct Coords;
struct Sdf;
struct int3;
struct float4;
struct Config;

struct WallQuants {
    float4 *pp; /* particle positions xyzo xyzo ... */
    int n;      /* number of particles              */
    int3 L;     /* subdomain size                   */
};

struct WallTicket;
struct WallRepulsePrm;


void wall_ini_quants(int3 L, WallQuants*);
void wall_ini_ticket(int3 L, WallTicket**);

void wall_fin_quants(WallQuants*);
void wall_fin_ticket(WallTicket*);

void wall_gen_quants(MPI_Comm, int numdensity, int maxn, const Sdf*, /* io */ int *o_n, Particle *o_pp, /**/ WallQuants*); // <1>
void wall_gen_ticket(const WallQuants*, WallTicket*); // <2>

void wall_strt_quants(MPI_Comm, const char *base, int maxn, WallQuants*); // <1>
void wall_strt_dump_templ(MPI_Comm, const char *base, const WallQuants*); // <2>

void wall_force(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                const WallTicket*, int n, const PaArray*, const FoArray*); // <1>

void wall_force_adhesion(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                         const WallTicket*, int n, const PaArray*, const FoArray*); // <2>

void wall_repulse(const Sdf*, const WallRepulsePrm*, long n, const PaArray*, const FoArray*); // <3>

void wall_repulse_prm_ini(float lambda, WallRepulsePrm**); // <1>
void wall_repulse_prm_ini_conf(const Config*, const char *ns, WallRepulsePrm**); // <2>
void wall_repulse_prm_fin(WallRepulsePrm*); // <3>

namespace wall_dev {
__global__ void particle2float4(const Particle *src, const int n, float4 *dst) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = src[pid];
    dst[pid] = make_float4(p.r[X], p.r[Y], p.r[Z], 0);
}

__global__ void float42particle(const float4 *src, const int n, Particle *dst) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    const float4 r = src[pid];
    Particle p;
    p.r[X] = r.x; p.r[Y] = r.y; p.r[Z] = r.z;
    p.v[X] = p.v[Y] = p.v[Z] = 0;
    dst[pid] = p;
}
}

struct WallTicket {
    RNDunif *rnd;        /* rng on host                                        */
    Clist *cells;        /* cell lists (always the same, no need to store map) */
    Texo<int> texstart;  /* texture of starts from clist                       */
    Texo<float4> texpp;  /* texture of particle positions                      */
};
void wall_ini_quants(int3 L, WallQuants *q) {
    q->n = 0;
    q->pp = NULL;
    q->L = L;
}

void wall_ini_ticket(int3 L, WallTicket **ti) {
    WallTicket *t;
    EMALLOC(1, ti);
    t = *ti;
    UC(rnd_ini(42, 42, 42, 42, /**/ &t->rnd));
    UC(clist_ini(L.x + 2 * XWM,
                 L.y + 2 * YWM,
                 L.z + 2 * ZWM, /**/ &t->cells));
}
void wall_fin_quants(WallQuants *q) {
    if (q->pp) Dfree(q->pp);
    q->n = 0;
}

void wall_fin_ticket(WallTicket *t) {
    UC(rnd_fin(t->rnd));
    UC(texo_destroy(&t->texstart));
    UC(texo_destroy(&t->texpp));
    UC(clist_fin(t->cells));
    EFREE(t);
}
#define CODE "wall"
#define PP CODE ".pp"

static void read(MPI_Comm comm, const char *base, int maxn, /**/ float4 *pp, int *n) {
    Particle *pphst, *ppdev;
    EMALLOC(maxn, &pphst);

    restart_read_pp(comm, base, PP, RESTART_TEMPL, n, pphst);

    if (*n) {
        Dalloc(&ppdev, maxn);
        cH2D(ppdev, pphst, *n);
        KL(wall_dev::particle2float4, (k_cnf(*n)), (ppdev, *n, /**/ pp));
        Dfree(ppdev);
    }
    EFREE(pphst);
}

static void write(MPI_Comm comm, const char *base, int n, const float4 *pp) {
    Particle *pphst, *ppdev;

    EMALLOC(n, &pphst);
    if (n) {
        Dalloc(&ppdev, n);
        KL(wall_dev::float42particle , (k_cnf(n)), (pp, n, /**/ ppdev));
        cD2H(pphst, ppdev, n);
        Dfree(ppdev);
    }
    restart_write_pp(comm, base, PP, RESTART_TEMPL, n, pphst);

    EFREE(pphst);
}

static void strt_quants(MPI_Comm comm, const char *base, int maxn, int *w_n, float4 **w_pp) {
    float4 * pptmp;
    Dalloc(&pptmp, maxn);
    read(comm, base, maxn, pptmp, w_n);

    if (*w_n) {
        Dalloc(w_pp, *w_n);
        cD2D(*w_pp, pptmp, *w_n);
    }
    Dfree(pptmp);
}

void wall_strt_quants(MPI_Comm comm, const char *base, int maxn, WallQuants *q) {
    strt_quants(comm, base, maxn, &q->n, &q->pp);
}

void wall_strt_dump_templ(MPI_Comm comm, const char *base, const WallQuants *q) {
    write(comm, base, q->n, q->pp);
}

#undef PP
#undef CODE
static void freeze0(MPI_Comm cart, int numdensity, int3 L, int maxn, const Sdf *qsdf, /*io*/ int *n, Particle *pp, /*o*/ int *w_n, Particle *dev, /*w*/ Particle *hst) {
    sdf_bulk_wall(qsdf, /*io*/ n, pp, /*o*/ w_n, hst); /* sort into bulk-frozen */
    msg_print("before exch: bulk/frozen : %d/%d", *n, *w_n);
    UC(wall_exch_pp(cart, numdensity, L, maxn, /*io*/ hst, w_n));
    cH2D(dev, hst, *w_n);
    msg_print("after  exch: bulk/frozen : %d/%d", *n, *w_n);
}

static void freeze(MPI_Comm cart, int numdensity, int3 L, int maxn, const Sdf *qsdf, /*io*/ int *n, Particle *pp, /*o*/ int *w_n, Particle *dev) {
    Particle *hst;
    EMALLOC(maxn, &hst);
    UC(freeze0(cart, numdensity, L, maxn, qsdf, /*io*/ n, pp, /*o*/ w_n, dev, /*w*/ hst));
    EFREE(hst);
}

static void gen_quants(MPI_Comm cart, int numdensity, int3 L, int maxn, const Sdf *qsdf, /**/ int *o_n, Particle *o_pp, int *w_n, float4 **w_pp) {
    Particle *frozen;
    Dalloc(&frozen, maxn);
    UC(freeze(cart, numdensity, L, maxn, qsdf, o_n, o_pp, w_n, frozen));
    msg_print("consolidating wall");
    Dalloc(w_pp, *w_n);
    KL(wall_dev::particle2float4, (k_cnf(*w_n)), (frozen, *w_n, /**/ *w_pp));

    Dfree(frozen);
    dSync();
}

void wall_gen_quants(MPI_Comm cart, int numdensity, int maxn, const Sdf *sdf, /* io */ int *o_n, Particle *o_pp, /**/ WallQuants *q) {
    int3 L = q->L;
    UC(gen_quants(cart, numdensity, L, maxn, sdf, o_n, o_pp, &q->n, &q->pp));
}

static void build_cells(const int n, float4 *pp4, Clist *cells, ClistMap *mcells) {
    if (n == 0) return;

    Particle *pp, *pp0;
    Dalloc(&pp,  n);
    Dalloc(&pp0, n);

    KL(wall_dev::float42particle, (k_cnf(n)), (pp4, n, /**/ pp));
    UC(clist_build(n, n, pp, /**/ pp0, cells, mcells));
    KL(wall_dev::particle2float4, (k_cnf(n)), (pp0, n, /**/ pp4));

    Dfree(pp);
    Dfree(pp0);
}

static void gen_ticket(const int w_n, float4 *w_pp, Clist *cells, Texo<int> *texstart, Texo<float4> *texpp) {
    ClistMap *mcells;
    UC(clist_ini_map(w_n, 1, cells, /**/ &mcells));
    UC(build_cells(w_n, /**/ w_pp, cells, mcells));

    UC(texo_setup(clists_get_n(cells), (int*) clists_get_ss(cells), texstart));
    UC(texo_setup(w_n, w_pp, texpp));
    UC(clist_fin_map(mcells));
}

void wall_gen_ticket(const WallQuants *q, WallTicket *t) {
    UC(gen_ticket(q->n, q->pp, t->cells, &t->texstart, &t->texpp));
}
static WallForce get_wa(const Sdf *sdf, const WallQuants *q, const WallTicket *t, int) {
    WallForce wa; /* local wall data */

    sdf_get_view(sdf, &wa.sdf_v);
    wa.start  = t->texstart;
    wa.pp     = t->texpp;
    wa.n      = q->n;
    wa.L      = q->L;

    return wa;
}

void wall_force(const PairParams *params, const WvelStep *wv, const Coords *c, const Sdf *sdf,
                const WallQuants *q, const WallTicket *t, int n, const PaArray *parray, const FoArray *farray) {
    WallForce wa;
    wa = get_wa(sdf, q, t, n);

    UC(wall_force_apply(params, wv, c, parray, n, t->rnd, wa, /**/ farray));
}

void wall_force_adhesion(const PairParams *params, const WvelStep *wv, const Coords *c, const Sdf *sdf,
                         const WallQuants *q, const WallTicket *t, int n, const PaArray *parray, const FoArray *farray) {
    WallForce wa;
    wa = get_wa(sdf, q, t, n);

    UC(wall_force_adhesion_apply(params, wv, c, parray, n, t->rnd, wa, /**/ farray));
}

void wall_repulse(const Sdf *sdf, const WallRepulsePrm *wr, long n, const PaArray *pa, const FoArray *fa) {
    Sdf_v sdf_v;
    sdf_get_view(sdf, &sdf_v);
    UC(wall_force_repulse(sdf_v, *wr, n, pa, /**/ fa));
}
void wall_repulse_prm_ini(float lambda, WallRepulsePrm **wrp) {
    WallRepulsePrm *wr;
    EMALLOC(1, wrp);
    wr = *wrp;
    wr->l = lambda;
}

void wall_repulse_prm_ini_conf(const Config *cfg, const char *ns, WallRepulsePrm **wrp) {
    float lambda;
    UC(conf_lookup_float_ns(cfg, ns, "lambda", &lambda));
    UC(wall_repulse_prm_ini(lambda, wrp));
}

void wall_repulse_prm_fin(WallRepulsePrm *wr) {
    EFREE(wr);
}





// Grouped editable compilation unit for the complete SDF subsystem.


// Grouped editable compilation unit for SDF core and label handling.



#include <mpi.h>

#include <curand.h>

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


/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
/* B.1. Function Type Qualifiers */
/* B.2. Variable Type Qualifiers */
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_WALL_DEVICE_API_COMMON
#define UDX_WALL_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
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
#ifndef UDX_WALL_DEF_COMMON
#define UDX_WALL_DEF_COMMON
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
#endif
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}



/* TODO: header includes header */
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
#ifndef UDX_WALL_KL_COMMON
#define UDX_WALL_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif

namespace m {
int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Barrier(MPI_Comm comm);
int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int Cancel(MPI_Request *request);
int Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart);
int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int Comm_free(MPI_Comm *comm);
int Comm_rank(MPI_Comm comm, int *rank);
int Comm_size( MPI_Comm comm, int *size );
int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int File_close(MPI_File *fh);
int File_get_position(MPI_File fh, MPI_Offset *offset);
int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
int File_seek(MPI_File fh, MPI_Offset offset, int whence);
int File_set_size(MPI_File fh, MPI_Offset size);
int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
int Finalize(void);
int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);
int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
              int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
int Init(int *argc, char ***argv);
int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int Type_commit(MPI_Datatype *type);
int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);
int Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int Error_string(int errorcode, char *string, int *resultlen);

/* local */
bool is_master(MPI_Comm comm);
int is_success(int errorcode);
int is_pending(int errorcode);
int is_err_in_status(int errorcode);
int status2errcode(MPI_Status *status);
}

#ifndef UDX_WALL_TYPE_COMMON
#define UDX_WALL_TYPE_COMMON
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
#endif
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
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);

struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);

/* edges of the sub-domain in global coordinates */

int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);

/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);

struct Array3d;
void array3d_ini(Array3d**, size_t x, size_t y, size_t z);                // <1>
void array3d_fin(Array3d*);                                               // <2>
void array3d_copy(size_t x, size_t y, size_t z, float *D, /**/ Array3d*); // <3>
#ifndef UDX_WALL_TEX3D_TYPE_COMMON
#define UDX_WALL_TEX3D_TYPE_COMMON
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
#endif
struct Tex3d;
struct Tex3d_v;
struct Array3d;

void tex3d_ini(Tex3d**);
void tex3d_fin(Tex3d*);

void tex3d_copy(const Array3d*, /**/ Tex3d*);    // <1>
void tex3d_to_view(const Tex3d*, /**/ Tex3d_v*); // <2>

struct Coords;
struct Tform;
struct Field;

void field_ini(const char *path, /**/ Field**); // <1>
void field_fin(Field*);                         // <2>

void field_sample(const Field*, const Tform*, const int N[3], /**/ Field**); // <1>

void field_size(const Field*, /**/ int N[3]);       // <1>
void field_extend(const Field*, /**/ float ext[3]); // <2>
void field_data(const Field*, /**/ float**);        // <3>

void field_dump(const Field*, const Coords*, MPI_Comm cart); // <2>
void field_scale(Field*, float scale); // <3>
struct Sdf;
struct WvelStep;
struct Coords;
struct Particle;

void bounce_back(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp);

#ifndef UDX_WALL_LABEL_COMMON
#define UDX_WALL_LABEL_COMMON
enum {
    LABEL_BULK,  /* remains in bulk */
    LABEL_WALL,  /* becomes wall particle */
    LABEL_DEEP   /* deep inside the wall */
};

void wall_label_dev(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <1>
void wall_label_hst(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <2>
#endif

#ifndef UDX_WALL_TFORM_TYPE_COMMON
#define UDX_WALL_TFORM_TYPE_COMMON
struct Tform_v { float o[3], s[3]; };
#endif
struct Tform;
struct Tform_v;

void tform_ini(Tform**);
void tform_fin(Tform*);

void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform*); // <1>
void tform_chain(const Tform *A, const Tform *B, /**/ Tform *C); // <2>
void tform_to_grid(const float lo[3], const float hi[3], const int n[3], /**/ Tform*); // <3>
void tform_from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform*); // <4>
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform*); // <5>


void tform_convert(const Tform*, const float a0[3], /**/ float a1[3]);

void tform_to_view(const Tform*, /**/ Tform_v*);

void tform_log(Tform*);
void tform_dump(Tform*, FILE*);
#ifndef UDX_WALL_TFORM_DEV_COMMON
#define UDX_WALL_TFORM_DEV_COMMON
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
#endif
struct Tform;
struct Coords;

/* T: texture size, N: sdf file size, M: wall margin */
void tex2sdf_ini(const Coords*, const int T[3], const int N[3], const int M[3], /**/ Tform*); // <1>
void out2sdf_ini(const Coords*, const int N[3], /**/ Tform*); // <2>
void sub2tex_ini(const Coords*, const int T[3], const int M[3], /**/ Tform*); // <3>

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
#define _S_ static __device__
#define _I_ static __device__

template <typename T>
_I_ T warpReduceSum(T v)  // <1>
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

_I_ float2 warpReduceSum(float2 val)  // <2>
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

_I_ float3 warpReduceSum(float3 val)  // <3>
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


#ifndef UDX_WALL_SDF_TYPE_COMMON
#define UDX_WALL_SDF_TYPE_COMMON
/* view */
struct Sdf_v {
    Tex3d_v tex;
    Tform_v t;
    float cheap_threshold;
};
#endif
struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);

void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*); // <1>
void sdf_get_view(const Sdf*, /**/ Sdf_v*);                  // <2>

void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp); // <1>
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);          // <2>
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);                          // <3>

void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);

#ifndef UDX_WALL_SDF_DEV_COMMON
#define UDX_WALL_SDF_DEV_COMMON
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

_I_ float3 sdf_ugrad(const Sdf_v *texsdf, const float3 *r) // <1>
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

_I_ bool sdf_far(const Sdf_v *sdf, float x, float y, float z) // <2>
{
    return cheap_sdf(sdf, x, y, z) <= sdf->cheap_threshold - 1 ;
}

_I_ float sdf_eval(const Sdf_v *sdf0, float x, float y, float z)  // <3>
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
#endif

namespace sdf_dev {
static __device__ float3 get_r(int3 L, const float *U) {
    enum {X, Y, Z};
    float3 r;
    r.x = L.x * (U[X] - 0.5);
    r.y = L.y * (U[Y] - 0.5);
    r.z = L.z * (U[Z] - 0.5);
    return r;
}

__global__ void count_inside(Sdf_v sv, int3 L, int n, const float *UU, int *counts) {
    int i, c, chunkid, warpmaster;
    float3 r;
    float s;

    c = 0;
    warpmaster = (0 == threadIdx.x % warpSize);
    i   = threadIdx.x + blockIdx.x * blockDim.x;
    chunkid = blockDim.x;

    if (i < n) {
        r = get_r(L, UU + 3 * i);
        s = sdf_eval(&sv, r.x, r.y, r.z);
        c = (s <= 0.f);
    }

    c = warpReduceSum(c);

    if (warpmaster)
        atomicAdd(&counts[chunkid], c);
}
}

struct Array3d;
struct Tform;
struct Sdf {
    Array3d   *arr;
    Tex3d     *tex;
    Tform     *t;
    float cheap_threshold;

    int3 Lte; /* texture grid size */
};
static void gen0(const Coords *coords, Field *F, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    int N[3];
    float *D;
    field_size(F, /**/  N);
    field_data(F, /**/ &D);
    UC(array3d_copy(N[X], N[Y], N[Z], D, /**/ sdf->arr));
    UC(tex3d_ini(&sdf->tex));
    UC(tex3d_copy(sdf->arr, /**/ sdf->tex));
    UC(sub2tex_ini(coords, N, M, /**/ sdf->t));
}

static void gen1(const Coords *coords, const int T[3], const Field *F0, /**/ Sdf *sdf) {
    Tform *t;
    int N[3];
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    Field *F1;

    UC(tform_ini(&t));
    field_size(F0, /**/ N);
    UC(tex2sdf_ini(coords, T, N, M, /**/ t));
    UC(field_sample(F0, t, T, /**/ &F1));
    UC(gen0(coords, F1, sdf));

    UC(field_fin(F1));
    UC(tform_fin(t));
}

static void gen2(const Coords *coords, const Field *F, /**/ Sdf *sdf) {
    int3 Lte = sdf->Lte;
    int T[] = {Lte.x, Lte.y, Lte.z};
    UC(gen1(coords, T, F, /**/ sdf));
}

static void gen3(const Coords *coords, bool dump, MPI_Comm cart, Field *F, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    float sc, G, ext[3];
    G = xdomain(coords);
    UC(field_extend(F, /**/ ext));
    sc = G / ext[X];
    UC(field_scale(F, sc));
    if (dump)
        UC(field_dump(F, coords, cart));
    UC(gen2(coords, F, /**/ sdf));
}

void sdf_gen(const Coords *coords, MPI_Comm cart, bool dump, Sdf *sdf) {
    enum {X, Y, Z};
    const char *f = "sdf.dat";
    Field *F;
    UC(field_ini(f, &F));
    UC(gen3(coords, dump, cart, F, /**/ sdf));
    UC(field_fin(F));
}
/* sort solvent particle into remaining in solvent and turning into wall according to keys (all on hst) */
static void split_wall_solvent(const int *keys, /*io*/ int *s_n, Particle *s_pp, /**/ int *w_n, Particle *w_pp) {
    int n = *s_n;
    Particle p;
    int k, ia = 0, is = 0, iw = 0; /* all, solvent, wall */

    for (ia = 0; ia < n; ++ia) {
        k = keys[ia];
        p = s_pp[ia];

        if      (k == LABEL_BULK) s_pp[is++] = p;
        else if (k == LABEL_WALL) w_pp[iw++] = p;
    }
    *s_n = is;
    *w_n = iw;
}

/* sort solvent particle (dev) into remaining in solvent (dev) and turning into wall (hst)*/
void sdf_bulk_wall(const Sdf *sdf, /*io*/ int* s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp) {
    int n = *s_n, *labels;
    Particle *s_pp_hst;
    EMALLOC(n, &s_pp_hst);
    EMALLOC(n, &labels);

    UC(wall_label_hst(sdf, n, s_pp, labels));
    cD2H(s_pp_hst, s_pp, n);

    UC(split_wall_solvent(labels, /*io*/ s_n, s_pp_hst, /**/ w_n, w_pp));
    cH2D(s_pp, s_pp_hst, *s_n);

    EFREE(s_pp_hst);
    EFREE(labels);
}

/* bulk predicate : is in bulk? */
static bool bulkp(int *keys, int i) { return keys[i] == LABEL_BULK; }

static int who_stays0(int *keys, int nc, int nv, /*o*/ int *stay) {
    int c, v;  /* cell and vertex */
    int s = 0; /* how many stays? */
    for (c = 0; c < nc; ++c) {
        v = 0;
        while (v  < nv && bulkp(keys, v + nv * c)) v++;
        if    (v == nv) stay[s++] = c;
    }
    return s;
}

int sdf_who_stays(const Sdf *sdf, int n, const Particle *pp, int nc, int nv, /**/ int *stay) {
    int *labels;
    EMALLOC(n, &labels);
    UC(wall_label_hst(sdf, n, pp, /**/ labels));
    nc = who_stays0(labels, nc, nv, /**/ stay);
    EFREE(labels);
    return nc;
}
static int3 texture_grid_size(int3 L, int3 M) {
    enum {PAD = 16};
    int3 T;
    int x, y, z; /* grid extents in real space */
    int ty, tz;

    x = L.x + 2 * M.x;
    y = L.y + 2 * M.y;
    z = L.z + 2 * M.z;

    T.x = PAD * PAD;
    ty = ceiln(y * T.x, x);
    tz = ceiln(z * T.x, x);
    T.y = PAD * ceiln(ty, PAD);
    T.z = PAD * ceiln(tz, PAD);

    return T;
};

void sdf_ini(int3 L, Sdf **pq) {
    Sdf *q;
    int3 Lte, M;
    M = make_int3(XWM, YWM, ZWM);
    Lte = texture_grid_size(L, M);
    EMALLOC(1, &q);
    UC(array3d_ini(&q->arr, Lte.x, Lte.y, Lte.z));
    UC(  tform_ini(&q->t));

    q->Lte = Lte;
    q->cheap_threshold = - sqrt(3.f) * ((float) (L.x + 2*M.x) / (float)Lte.x);

    *pq = q;
}

void sdf_fin(Sdf *q) {
    UC(array3d_fin(q->arr));
    UC(  tex3d_fin(q->tex));
    UC(  tform_fin(q->t));
    EFREE(q);
}

void sdf_bounce(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    UC(bounce_back(dt, wv, c, sdf, n, /**/ pp));
}


void sdf_get_view(const Sdf *q, /**/ Sdf_v *v) {
    tex3d_to_view(q->tex, &v->tex);
    tform_to_view(q->t  , &v->t);
    v->cheap_threshold = q->cheap_threshold;
}

enum {NTHREADS = 128};

static void gen_rnd(long n, float *UU) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, UU, n);
    curandDestroyGenerator(gen);
}

static void chunk_counts(int3 L, const Sdf *sdf, int nsamples, int nchunks, int *counts_hst) {
    int *counts_dev;
    Sdf_v view;
    float *UU;
    sdf_get_view(sdf, &view);

    Dalloc(&UU, 3 * nsamples);
    Dalloc(&counts_dev, nchunks);
    DzeroA(counts_dev, nchunks);

    gen_rnd(3 * nsamples, UU);

    KL(sdf_dev::count_inside,
       (nchunks, NTHREADS),
       (view, L, nsamples, UU, /**/ counts_dev));

    cD2H(counts_hst, counts_dev, nchunks);
    Dfree(counts_dev);
    Dfree(UU);
}

static long reduce(int n, int *d) {
    long c, i;
    for (i = c = 0; i < n; ++i) c += d[i];
    return c;
}

static long subdomain_counts(int3 L, const Sdf *sdf, long nsamples) {
    int *counts_hst;
    long nin;
    int nchunks;

    nchunks = ceiln(nsamples, NTHREADS);
    EMALLOC(nchunks, &counts_hst);

    chunk_counts(L, sdf, nsamples, nchunks, counts_hst);
    nin = reduce(nchunks, counts_hst);

    EFREE(counts_hst);
    return nin;
}

static double subdomain_volume(int3 L, long nin, long nsamples) {
    double V = (double) nin / (double) nsamples;
    V *= L.x * L.y * L.z;
    return V;
}

double sdf_compute_volume(MPI_Comm comm, int3 L, const Sdf *sdf, long nsamples) {
    double loc, tot = 0;
    long nin;

    nin = subdomain_counts(L, sdf, nsamples);
    loc = subdomain_volume(L, nin, nsamples);

    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, comm));

    return tot;
}





#include <assert.h>
#include <mpi.h>

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


#ifndef UDX_WALL_DEF_COMMON
#define UDX_WALL_DEF_COMMON
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
#endif
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}



/* TODO: header includes header */
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
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);

/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
/* B.1. Function Type Qualifiers */
/* B.2. Variable Type Qualifiers */
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_WALL_DEVICE_API_COMMON
#define UDX_WALL_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif

#ifndef UDX_WALL_TYPE_COMMON
#define UDX_WALL_TYPE_COMMON
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
#endif
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

#ifndef UDX_WALL_KL_COMMON
#define UDX_WALL_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif

#ifndef UDX_WALL_TFORM_TYPE_COMMON
#define UDX_WALL_TFORM_TYPE_COMMON
struct Tform_v { float o[3], s[3]; };
#endif
#ifndef UDX_WALL_TFORM_DEV_COMMON
#define UDX_WALL_TFORM_DEV_COMMON
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
#endif
#ifndef UDX_WALL_TEX3D_TYPE_COMMON
#define UDX_WALL_TEX3D_TYPE_COMMON
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
#endif
#ifndef UDX_WALL_SDF_TYPE_COMMON
#define UDX_WALL_SDF_TYPE_COMMON
/* view */
struct Sdf_v {
    Tex3d_v tex;
    Tform_v t;
    float cheap_threshold;
};
#endif
struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);

void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*); // <1>
void sdf_get_view(const Sdf*, /**/ Sdf_v*);                  // <2>

void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp); // <1>
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);          // <2>
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);                          // <3>

void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);
#ifndef UDX_WALL_SDF_DEV_COMMON
#define UDX_WALL_SDF_DEV_COMMON
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

_I_ float3 sdf_ugrad(const Sdf_v *texsdf, const float3 *r) // <1>
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

_I_ bool sdf_far(const Sdf_v *sdf, float x, float y, float z) // <2>
{
    return cheap_sdf(sdf, x, y, z) <= sdf->cheap_threshold - 1 ;
}

_I_ float sdf_eval(const Sdf_v *sdf0, float x, float y, float z)  // <3>
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
#endif

#ifndef UDX_WALL_LABEL_COMMON
#define UDX_WALL_LABEL_COMMON
enum {
    LABEL_BULK,  /* remains in bulk */
    LABEL_WALL,  /* becomes wall particle */
    LABEL_DEEP   /* deep inside the wall */
};

void wall_label_dev(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <1>
void wall_label_hst(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <2>
#endif

namespace sdf_label_dev {
static __global__ void main(Sdf_v sdf_v, int n, const Particle *pp, /**/ int *labels) {
    enum {X, Y, Z};
    int pid;
    Particle p;
    float s;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    p = pp[pid];
    s = sdf_eval(&sdf_v, p.r[X], p.r[Y], p.r[Z]);
    labels[pid] =
        s >  2 ? LABEL_DEEP :
        s >= 0 ? LABEL_WALL :
                 LABEL_BULK;
}
}
void wall_label_dev(const Sdf *sdf, int n, const Particle *pp, /**/ int *labels) {
    Sdf_v sdf_v;
    sdf_get_view(sdf, &sdf_v);
    KL(sdf_label_dev::main,(k_cnf(n)), (sdf_v, n, pp, labels));
}

void wall_label_hst(const Sdf *sdf, int n, const Particle *pp, /**/ int *hst) {
    int *dev;
    Sdf_v sdf_v;
    Dalloc(&dev, n);
    sdf_get_view(sdf, &sdf_v);
    KL(sdf_label_dev::main,(k_cnf(n)), (sdf_v, n, pp, dev));
    cD2H(hst, dev, n);
    Dfree(dev);
}





// Grouped editable compilation unit for SDF transforms, velocity, and bounce.


// Grouped editable compilation unit for SDF transforms and wall velocity.


// Grouped editable compilation unit for SDF grid storage and transforms.



#include <stdio.h>

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


#ifndef UDX_WALL_DEVICE_API_COMMON
#define UDX_WALL_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif

void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
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
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}



/* TODO: header includes header */
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

#ifndef UDX_WALL_SDF_ARRAY3D_TYPE
#define UDX_WALL_SDF_ARRAY3D_TYPE
struct Array3d {
    cudaArray_t a;  // <1>
    size_t x, y, z; // <2>
};
#endif
struct Array3d;
void array3d_ini(Array3d**, size_t x, size_t y, size_t z);                // <1>
void array3d_fin(Array3d*);                                               // <2>
void array3d_copy(size_t x, size_t y, size_t z, float *D, /**/ Array3d*); // <3>

void array3d_ini(Array3d **pq, size_t x, size_t y, size_t z) {
    Array3d *q;
    cudaChannelFormatDesc fmt;
    EMALLOC(1, &q);

    fmt = cudaCreateChannelDesc<float>();
    msg_print("alloc cuda 3D Array: %ld %ld %ld", x, y, z);
    CC(cudaMalloc3DArray(&q->a, &fmt, make_cudaExtent(x, y, z)));
    q->x = x; q->y = y; q->z = z;

    *pq = q;
}

void array3d_fin(Array3d *q) {
    CC(cudaFreeArray(q->a));
    EFREE(q);
}

static int good(size_t x, size_t y, size_t z, Array3d *q) {
    return x == q->x && y == q->y && z == q->z;
}

void array3d_copy(size_t x, size_t y, size_t z, float *D, /**/ Array3d *q) {
    cudaMemcpy3DParms copyParams;
    if (!good(x, y, z, q))
        ERR("wrong size: %ld, %ld, %ld   !=   %ld, %ld, %ld",
            x, y, z, q->x, q->y, q->z);
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.srcPtr = make_cudaPitchedPtr((void*)D, x*sizeof(float), x, y);
    copyParams.dstArray = q->a;
    copyParams.extent = make_cudaExtent(x, y, z);
    copyParams.kind = cudaMemcpyHostToDevice;
    CC(cudaMemcpy3D(&copyParams));
}





#include <stdio.h>
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
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);

/* edges of the sub-domain in global coordinates */

int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);

/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);
struct Tform;
struct Tform_v;

void tform_ini(Tform**);
void tform_fin(Tform*);

void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform*); // <1>
void tform_chain(const Tform *A, const Tform *B, /**/ Tform *C); // <2>
void tform_to_grid(const float lo[3], const float hi[3], const int n[3], /**/ Tform*); // <3>
void tform_from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform*); // <4>
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform*); // <5>


void tform_convert(const Tform*, const float a0[3], /**/ float a1[3]);

void tform_to_view(const Tform*, /**/ Tform_v*);

void tform_log(Tform*);
void tform_dump(Tform*, FILE*);

struct Tform;
struct Coords;

/* T: texture size, N: sdf file size, M: wall margin */
void tex2sdf_ini(const Coords*, const int T[3], const int N[3], const int M[3], /**/ Tform*); // <1>
void out2sdf_ini(const Coords*, const int N[3], /**/ Tform*); // <2>
void sub2tex_ini(const Coords*, const int T[3], const int M[3], /**/ Tform*); // <3>


struct TGrid {
    float lo[3], hi[3];
    int n[3];
};

enum {OK, BAD};
static int goodp(const int n[3]) {
    enum {X, Y, Z};
    const int H = 1000000;
    int cl, ch;
    cl = n[X] > 0 && n[Y] > 0 && n[Z] > 0;
    ch = n[X] < H && n[Y] < H && n[Z] < H;
    return (cl && ch) ? OK : BAD;
}


static void grid2grid(TGrid *from, TGrid *to, /**/ Tform *t) {
    tform_grid2grid(from->lo, from->hi, from->n,
                    to->lo, to->hi,   to->n, /**/ t);
}

static void sdf_ini(const Coords *c, const int N[3], /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = t->lo[Y] = t->lo[Z] = 0;
    t->hi[X] = xdomain(c);
    t->hi[Y] = ydomain(c);
    t->hi[Z] = zdomain(c);

    t->n[X] = N[X]; t->n[Y] = N[Y]; t->n[Z] = N[Z];
}

static void tex_ini(const Coords *c, const int T[3], const int M[3], /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(c) - M[X];
    t->lo[Y] = ylo(c) - M[Y];
    t->lo[Z] = zlo(c) - M[Z];
    t->hi[X] = xhi(c) + M[X];
    t->hi[Y] = yhi(c) + M[Y];
    t->hi[Z] = zhi(c) + M[Z];

    t->n[X] = T[X]; t->n[Y] = T[Y]; t->n[Z] = T[Z];
}

static void out_ini(const Coords *c, /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(c);
    t->lo[Y] = ylo(c);
    t->lo[Z] = zlo(c);

    t->hi[X] = xhi(c);
    t->hi[Y] = yhi(c);
    t->hi[Z] = zhi(c);

    t->n[X] = xs(c);
    t->n[Y] = ys(c);
    t->n[Z] = zs(c);
}

static void sub_ini(const Coords *c, /**/ Tform *t) {
    enum {X, Y, Z};
    float a0[3], a1[3], b0[3], b1[3];
    a0[X] = -xs(c)/2; a0[Y] = -ys(c)/2; a0[Z] = -zs(c)/2;
    b0[X] =  xs(c)/2; b0[Y] =  ys(c)/2; b0[Z] =  zs(c)/2;

    a1[X] = xlo(c); a1[Y] = ylo(c); a1[Z] = zlo(c);
    b1[X] = xhi(c); b1[Y] = yhi(c); b1[Z] = zhi(c);
    UC(tform_vector(a0, a1,    b0, b1, /**/ t));
}

void tex2sdf_ini(const Coords *c,
                 const int T[3], const int N[3], const int M[3],
                 /**/ Tform *t) {
    enum {X, Y, Z};
    TGrid tex, sdf;

    if (goodp(T) == BAD) ERR("bad T = [%d %d %d]", T[X], T[Y], T[Z]);
    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    if (goodp(M) == BAD) ERR("bad M = [%d %d %d]", M[X], M[Y], M[Z]);

    tex_ini(c, T, M, /**/ &tex);
    sdf_ini(c, N, /**/ &sdf);

    UC(grid2grid(&tex, &sdf, /**/ t));
}

void out2sdf_ini(const Coords *c, const int N[3], /**/ Tform* t) {
    enum {X, Y, Z};
    TGrid sub, sdf;
    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    out_ini(c, /**/ &sub);
    sdf_ini(c, N, /**/ &sdf);
    UC(grid2grid(&sub, &sdf, /**/ t));
}

void sub2tex_ini(const Coords *c, const int T[3], const int M[3], /**/ Tform *t) {
    enum {X, Y, Z, D};
    float lo[D], hi[D];

    if (goodp(T) == BAD) ERR("bad T = [%d %d %d]", T[X], T[Y], T[Z]);
    if (goodp(M) == BAD) ERR("bad M = [%d %d %d]", M[X], M[Y], M[Z]);

    lo[X] = -xs(c)/2 - M[X];
    lo[Y] = -ys(c)/2 - M[Y];
    lo[Z] = -zs(c)/2 - M[Z];

    hi[X] = xs(c)/2 + M[X];
    hi[Y] = ys(c)/2 + M[Y];
    hi[Z] = zs(c)/2 + M[Z];

    tform_to_grid(lo, hi, T, /**/ t);
}





// Grouped editable compilation unit for SDF sampling and wall velocity models.


// Grouped editable compilation unit for SDF field and texture sampling.



#include <mpi.h>
#include <math.h>
#include <vector_types.h>

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
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
void grid_write(int3 subGrid, int3 subDomain, MPI_Comm, const char *path, int ncmp, const float **data, const char **names);
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);

/* edges of the sub-domain in global coordinates */

int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);

/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);

struct Tform;
struct Tform_v;

void tform_ini(Tform**);
void tform_fin(Tform*);

void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform*); // <1>
void tform_chain(const Tform *A, const Tform *B, /**/ Tform *C); // <2>
void tform_to_grid(const float lo[3], const float hi[3], const int n[3], /**/ Tform*); // <3>
void tform_from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform*); // <4>
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform*); // <5>


void tform_convert(const Tform*, const float a0[3], /**/ float a1[3]);

void tform_to_view(const Tform*, /**/ Tform_v*);

void tform_log(Tform*);
void tform_dump(Tform*, FILE*);
struct Tform;
struct Coords;

/* T: texture size, N: sdf file size, M: wall margin */
void tex2sdf_ini(const Coords*, const int T[3], const int N[3], const int M[3], /**/ Tform*); // <1>
void out2sdf_ini(const Coords*, const int N[3], /**/ Tform*); // <2>
void sub2tex_ini(const Coords*, const int T[3], const int M[3], /**/ Tform*); // <3>


struct Coords;
struct Tform;
struct Field;

void field_ini(const char *path, /**/ Field**); // <1>
void field_fin(Field*);                         // <2>

void field_sample(const Field*, const Tform*, const int N[3], /**/ Field**); // <1>

void field_size(const Field*, /**/ int N[3]);       // <1>
void field_extend(const Field*, /**/ float ext[3]); // <2>
void field_data(const Field*, /**/ float**);        // <3>

void field_dump(const Field*, const Coords*, MPI_Comm cart); // <2>
void field_scale(Field*, float scale); // <3>

struct Field {  /* size, extension, and data */
    int N[3];
    float ext[3];
    float *D;
};
static void read_extents(FILE *f, /**/ float *x, float *y, float *z) {
    int r;
    char s[2048];
    UC(efgets(s, sizeof(s), f));
    r = sscanf(s, "%f %f %f", x, y, z);
    if (r != 3)
        ERR("expect three floats, got '%s'", s);
}

static void read_grid_size(FILE *f, /**/ int *x, int *y, int *z) {
    int r;
    char s[2048];
    UC(efgets(s, sizeof(s), f));
    r = sscanf(s, "%d %d %d", x, y, z);
    if (r != 3)
        ERR("expect three integers, got '%s'", s);
}

void field_ini(const char *path, /**/ Field **pq) {
    enum {X, Y, Z};
    Field *q;
    int n, *N;
    float *ext, *D;
    FILE *f;

    EMALLOC(1, &q);
    N = q->N; ext = q->ext;
    msg_print("reading '%s'", path);
    UC(efopen(path, "r", /**/ &f));

    read_extents  (f, /**/ &ext[X], &ext[Y], &ext[Z]);
    read_grid_size(f, /**/ &N[X],     &N[Y],   &N[Z]);

    n = N[X]*N[Y]*N[Z];
    EMALLOC(n, &q->D);
    D = q->D;
    UC(efread(D, sizeof(D[0]), n, f));
    msg_print("size:   %d %d %d", N[X], N[Y], N[Z]);
    msg_print("extend: %g %g %g", ext[X], ext[Y], ext[Z]);
    msg_print("value: D[0,n-1] = %g %g", D[0], D[n-1]);

    *pq = q;
}
void field_fin(Field *q) {
    EFREE(q->D);
    EFREE(q);
}
struct Fi { /* field */
    const Tform *t;
    const int *n;
    float *D;
};

static void fi_ini(const Tform *t, const int *n, float *D, /**/ Fi *fi) {
    fi->t = t; fi->n = n; fi->D = D;
}

static void fi_r(const Fi *fi, int ix, int iy, int iz, /**/ float *r) {
    enum {X, Y, Z};
    const Tform *t;
    t = fi->t;
    float a[3] = {(float)ix, (float)iy, (float)iz};
    tform_convert(t, a, /**/ r);
}

static void fi_set(const Fi *fi, int ix, int iy, int iz, float v) {
    enum {X, Y};
    float *D;
    const int *n;
    int i;
    D = fi->D; n = fi->n;
    i = ix + n[X] * (iy + n[Y] * iz);
    D[i] = v;
}

static float spl(float x) { /* b-spline (see poc/spline/main.mac) */
    return
        x <= 0 ? 0.0 :
        x <= 1 ? x*x*x/6 :
        x <= 2 ? (x*((12-3*x)*x-12)+4)/6 :
        x <= 3 ? (x*(x*(3*x-24)+60)-44)/6 :
        x <= 4 ? (x*((12-x)*x-48)+64)/6 :
        0.0;
}

static float get(const int N[3], const float *D, const float *r) {
    enum {X, Y, Z};
#define DDD(ix, iy, iz) (D [ix + N[X] * (iy + N[Y] * iz)])
    int i, c, sx, sy, sz, anchor[3], g[3];
    float val, s, w[3][4], tmp[4][4], partial[4];

    for (c = 0; c < 3; ++c)
        anchor[c] = (int)floor(r[c]);

    for (c = 0; c < 3; ++c)
        for (i = 0; i < 4; ++i)
            w[c][i] = spl(r[c] - (anchor[c] - 1 + i) + 2);

    for (sz = 0; sz < 4; ++sz) {
        for (sy = 0; sy < 4; ++sy) {
            s = 0;
            for (sx = 0; sx < 4; ++sx) {
                int l[3] = {sx, sy, sz};
                for (c = 0; c < 3; ++c)
                    g[c] = (l[c] - 1 + anchor[c] + N[c]) % N[c];
                s += w[0][sx] * DDD(g[X], g[Y], g[Z]);
            }
            tmp[sz][sy] = s;
        }
    }
    for (sz = 0; sz < 4; ++sz) {
        s = 0;
        for (sy = 0; sy < 4; ++sy) s += w[1][sy] * tmp[sz][sy];
        partial[sz] = s;
    }
    val = 0;
    for (sz = 0; sz < 4; ++sz) val += w[2][sz] * partial[sz];
    return val;
}

static void sample(const Tform *t, const int N0[3], const float *D0, const int N1[3], float *D1) {
    enum {X, Y, Z};
    int ix, iy, iz;
    float val, r[3];
    Fi fi;
    fi_ini(t, N1, D1, /**/ &fi);
    for (iz = 0; iz < N1[Z]; ++iz) {
        for (iy = 0; iy < N1[Y]; ++iy) {
            for (ix = 0; ix < N1[X]; ++ix) {
                fi_r(&fi, ix, iy, iz, /**/ r);
                val = get(N0, D0, r);
                fi_set(&fi, ix, iy, iz, val);
            }
        }
    }
}
static void dump0(const Coords *coords, const int N0[3], const float *D0, /**/ float *D1) {
    int L[3] = {xs(coords), ys(coords), zs(coords)};
    Tform *t;
    UC(tform_ini(&t));
    UC(out2sdf_ini(coords, N0, /**/ t));
    UC(sample(t, N0, D0,   L, /**/ D1));
    UC(tform_fin(t));
}

static void dump1(const Coords *coords, MPI_Comm cart, const int N[3], const float *D, /*w*/ float *W) {
    int3 L = subdomain(coords);
    const char *names[] = {"wall"};
    const float *data[] = {W};

    UC(dump0(coords, N, D, /**/ W));
    UC(grid_write(L, L, cart, DUMP_BASE "/raw/wall.raw", 1, data, names));
}

static void dump(const Coords *coords, MPI_Comm cart, const int N[], const float *D) {
    float *W;
    int ngrid = xs(coords) * ys(coords) * zs(coords);
    EMALLOC(ngrid, &W);
    UC(dump1(coords, cart, N, D, /*w*/ W));
    EFREE(W);
}

void field_size(const Field *q, /**/ int N[3]) {
    enum {X, Y, Z};
    N[X] = q->N[X]; N[Y] = q->N[Y]; N[Z] = q->N[Z];
}
void field_extend(const Field *q, /**/ float ext[3]) {
    enum {X, Y, Z};
    ext[X] = q->ext[X]; ext[Y] = q->ext[Y]; ext[Z] = q->ext[Z];
}
void field_data(const Field *q, /**/ float **pD) {
    *pD = q->D;
}
void field_scale(Field *q, float scale) {
    enum {X, Y, Z};
    float *D;
    int i, n, *N;
    N = q->N; D = q->D;
    n = N[X]*N[Y]*N[Z];
    for (i = 0; i < n; i++) D[i] *= scale;
}

void field_sample(const Field *F, const Tform *t, const int N1[3], /**/ Field **pq) {
    enum {X, Y, Z};
    int n;
    Field *q;
    EMALLOC(1, &q);
    n = N1[X]*N1[Y]*N1[Z];
    EMALLOC(n, &q->D);
    sample(t, F->N, F->D, N1, /**/ q->D);
    q->N[X] = N1[X];
    q->N[Y] = N1[Y];
    q->N[Z] = N1[Z];
    /* q->ext ?  */
    *pq = q;
}

void field_dump(const Field *q, const Coords *c, MPI_Comm cart) {
    dump(c, cart, q->N, q->D);
}





#include <stdio.h>

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


#ifndef UDX_WALL_DEVICE_API_COMMON
#define UDX_WALL_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif

void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
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
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}



/* TODO: header includes header */

#ifndef UDX_WALL_SDF_ARRAY3D_TYPE
#define UDX_WALL_SDF_ARRAY3D_TYPE
struct Array3d {
    cudaArray_t a;  // <1>
    size_t x, y, z; // <2>
};
#endif

#ifndef UDX_WALL_TEX3D_TYPE_COMMON
#define UDX_WALL_TEX3D_TYPE_COMMON
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
#endif
struct Tex3d;
struct Tex3d_v;
struct Array3d;

void tex3d_ini(Tex3d**);
void tex3d_fin(Tex3d*);

void tex3d_copy(const Array3d*, /**/ Tex3d*);    // <1>
void tex3d_to_view(const Tex3d*, /**/ Tex3d_v*); // <2>

void tex3d_ini(Tex3d **pq) {
    Tex3d *q;
    EMALLOC(1, &q);
    *pq = q;
}

void tex3d_fin(Tex3d *q) {
    CC(cudaDestroyTextureObject(q->t));
    EFREE(q);
}

void tex3d_copy(const Array3d *a, /**/ Tex3d *t) {
    cudaResourceDesc resD;
    cudaTextureDesc  texD;

    memset(&resD, 0, sizeof(resD));
    resD.resType = cudaResourceTypeArray;
    resD.res.array.array = a->a;

    memset(&texD, 0, sizeof(texD));
    texD.normalizedCoords = 0;
    texD.filterMode = cudaFilterModePoint;
    texD.mipmapFilterMode = cudaFilterModePoint;
    texD.addressMode[0] = cudaAddressModeWrap;
    texD.addressMode[1] = cudaAddressModeWrap;
    texD.addressMode[2] = cudaAddressModeWrap;

    CC(cudaCreateTextureObject(&t->t, &resD, &texD, NULL));
}

void tex3d_to_view(const Tex3d *t, /**/ Tex3d_v *v) { v->t = t->t; }






#include <stdio.h>
#include <vector_types.h>

void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
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
struct Config;
struct float3;
struct int3;

void conf_ini(/**/ Config **c);
void conf_fin(/**/ Config *c);

void conf_read(int argc, char **argv, /**/ Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);     // <1>
void conf_write_history(const Config *cfg, FILE *stream); // <2>

struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

#ifndef UDX_WALL_WVEL_ENUM
#define UDX_WALL_WVEL_ENUM
enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};
#endif

void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);

void wvel_step_ini(WvelStep **wv);
void wvel_step_fin(WvelStep *wv);

void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
void wvel_set_timestep(float dt, Wvel *vw);

void wvel_set_conf(const Config *cfg, Wvel *vw);

void wvel_get_step(float t, const Wvel *wv, /**/ WvelStep *);

int wvel_get_type(const WvelStep *w);

void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v);

void wvel_set_conf(const Config *cfg, Wvel *vw) {
    const char *type;
    UC(conf_lookup_string(cfg, "wvel.type", &type));

    if (same_str(type, "constant")) {
        float3 u;
        UC(conf_lookup_float3(cfg, "wvel.u", &u));
        UC(wvel_set_cste(u, vw));
    }
    else if (same_str(type, "shear")) {
        float gdot;
        int vdir, gdir, half;
        UC(conf_lookup_float(cfg, "wvel.gdot", &gdot));
        UC(conf_lookup_int(cfg, "wvel.vdir", &vdir));
        UC(conf_lookup_int(cfg, "wvel.gdir", &gdir));
        UC(conf_lookup_int(cfg, "wvel.half", &half));
        UC(wvel_set_shear(gdot, vdir, gdir, half, vw));
    }
    else if (same_str(type, "shear sin")) {
        float gdot, w;
        int vdir, gdir, half;
        UC(conf_lookup_float(cfg, "wvel.gdot", &gdot));
        UC(conf_lookup_int(cfg, "wvel.vdir", &vdir));
        UC(conf_lookup_int(cfg, "wvel.gdir", &gdir));
        UC(conf_lookup_int(cfg, "wvel.half", &half));
        UC(conf_lookup_float(cfg, "wvel.w", &w));
        UC(wvel_set_shear_sin(gdot, vdir, gdir, half, w, vw));
    }
    else if (same_str(type, "hele shaw")) {
        float u, h;
        UC(conf_lookup_float(cfg, "wvel.u", &u));
        UC(conf_lookup_float(cfg, "wvel.h", &h));
        UC(wvel_set_hs(u, h, vw));
    }
    else {
        ERR("unknown type <%s>\n", type);
    }
}



#include <mpi.h>
#include <math.h>
#include <vector_types.h>
#include <assert.h>

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

void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);

/* edges of the sub-domain in global coordinates */

int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);

/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);

#ifndef UDX_WALL_WVEL_TYPE_COMMON
#define UDX_WALL_WVEL_TYPE_COMMON
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

#endif


/* data stored on the host                       */
/* (helpers to setup the views at each timestep) */

struct WvelCste {
    float3 u; // velocity amplitude
};

struct WvelShear {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // only half is moving ?
};

struct WvelShearSin : WvelShear {
    float w;        // frequency
};

struct WvelHS {
    float u; // radial max inflow
    float h; // height of the channel
};

enum {
    WALL_VEL_CSTE,
    WALL_VEL_SHEAR,
    WALL_VEL_SHEAR_SIN,
    WALL_VEL_HS,
};

union WvelPar {
    WvelCste cste;
    WvelShear shear;
    WvelShearSin shearsin;
    WvelHS hs;
};

/* main structure */

struct Wvel {
    WvelPar p;  /* parameters             */
    int type;
};

/* step */

union WvelPar_v {
    WvelCste_v cste;
    WvelShear_v shear;
    WvelHS_v hs;
};

/* device structure: to be passed to device code */

struct WvelStep {
    WvelPar_v p;
    int type;
};

void wvel_ini(Wvel **wv) {
    EMALLOC(1, wv);
}

void wvel_fin(Wvel *wv) {
    EFREE(wv);
}

void wvel_step_ini(WvelStep **wv) {
    EMALLOC(1, wv);
}

void wvel_step_fin(WvelStep *wv) {
    EFREE(wv);
}

void wvel_set_cste(float3 u, Wvel *vw) {
    WvelCste p;
    p.u = u;
    vw->type = WALL_VEL_CSTE;
    vw->p.cste = p;
}

void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw) {
    WvelShear p;
    p.gdot = gdot;
    p.vdir = vdir;
    p.gdir = gdir;
    p.half = half;

    vw->type = WALL_VEL_SHEAR;
    vw->p.shear = p;
}

void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw) {
    WvelShearSin p;
    p.gdot = gdot;
    p.vdir = vdir;
    p.gdir = gdir;
    p.half = half;
    p.w    = w;

    vw->type = WALL_VEL_SHEAR_SIN;
    vw->p.shearsin = p;
}

void wvel_set_hs(float u, float h, Wvel *vw) {
    WvelHS p;
    p.u = u;
    p.h = h;
    vw->type = WALL_VEL_HS;
    vw->p.hs = p;
}

static void set_step_cste(WvelCste p, WvelStep *wv) {
    wv->type = WALL_VEL_V_CSTE;
    wv->p.cste.u = p.u;
}

static void set_step_shear(WvelShear p, WvelStep *wv) {
    wv->type = WALL_VEL_V_SHEAR;
    wv->p.shear.gdot = p.gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
}

static void set_step_shear(float t, WvelShearSin p, WvelStep *wv) {
    float gdot;
    float w;
    wv->type = WALL_VEL_V_SHEAR;
    w = p.w;
    gdot = p.gdot * sin(w * t);

    wv->p.shear.gdot = gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
}

static void set_step_hs(WvelHS p, WvelStep *wv) {
    wv->type = WALL_VEL_V_HS;
    wv->p.hs.u = p.u;
    wv->p.hs.h = p.h;
}

void wvel_get_step(float t, const Wvel *wv, /**/ WvelStep *view) {
    switch (wv->type) {
    case WALL_VEL_CSTE:
        set_step_cste(wv->p.cste, /**/ view);
        break;
    case WALL_VEL_SHEAR:
        set_step_shear(wv->p.shear, /**/ view);
        break;
    case WALL_VEL_SHEAR_SIN:
        set_step_shear(t, wv->p.shearsin, /**/ view);
        break;
    case WALL_VEL_HS:
        set_step_hs(wv->p.hs, /**/ view);
        break;
    default:
        ERR("wrong type provided: <%d>", wv->type);
        break;
    };
}

int wvel_get_type(const WvelStep *w) {
    return w->type;
}

void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v) {
    *v = w->p.cste;
}

void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v) {
    *v = w->p.shear;
}

void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v) {
    *v = w->p.hs;
}







#include <stdio.h>
#include <assert.h>
#include <mpi.h>

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


void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}



/* TODO: header includes header */
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

/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
/* B.1. Function Type Qualifiers */
/* B.2. Variable Type Qualifiers */
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_WALL_DEVICE_API_COMMON
#define UDX_WALL_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif

#ifndef UDX_WALL_TYPE_COMMON
#define UDX_WALL_TYPE_COMMON
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
#endif /* UDX_WALL_TYPE_COMMON */
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

#ifndef UDX_WALL_KL_COMMON
#define UDX_WALL_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif /* UDX_WALL_KL_COMMON */

#ifndef UDX_WALL_COORDS_TYPE_COMMON
#define UDX_WALL_COORDS_TYPE_COMMON
struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
#endif /* UDX_WALL_COORDS_TYPE_COMMON */
#ifndef UDX_WALL_COORDS_DEV_COMMON
#define UDX_WALL_COORDS_DEV_COMMON
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

#endif /* UDX_WALL_COORDS_DEV_COMMON */
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);

/* edges of the sub-domain in global coordinates */

int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);

/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);

#ifndef UDX_WALL_WVEL_TYPE_COMMON
#define UDX_WALL_WVEL_TYPE_COMMON
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

#endif
struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

#ifndef UDX_WALL_WVEL_ENUM
#define UDX_WALL_WVEL_ENUM
enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};
#endif

void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);

void wvel_step_ini(WvelStep **wv);
void wvel_step_fin(WvelStep *wv);

void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
void wvel_set_timestep(float dt, Wvel *vw);

void wvel_set_conf(const Config *cfg, Wvel *vw);

void wvel_get_step(float t, const Wvel *wv, /**/ WvelStep *);

int wvel_get_type(const WvelStep *w);

void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v);
#ifndef UDX_WALL_WVEL_DEV_COMMON
#define UDX_WALL_WVEL_DEV_COMMON
static __device__ void wvel(WvelCste_v p, Coords_v c, float3 r, /**/ float3 *v)
{
    *v = p.u;
}

static __device__ void wvel(WvelShear_v p, Coords_v c, float3 r, /**/ float3 *v)
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

static __device__ void wvel(WvelHS_v p, Coords_v c, float3 r, /**/ float3 *v)
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

template <typename Wvel_v>
static __device__ void bounce_vel(Wvel_v wv, Coords_v c, float3 rw, /* io */ float3* v)
{
    float3 vw;
    wvel(wv, c, rw, /**/ &vw);
    v->x = 2 * vw.x - v->x;
    v->y = 2 * vw.y - v->y;
    v->z = 2 * vw.z - v->z;
}
#endif /* UDX_WALL_WVEL_DEV_COMMON */
#ifndef UDX_WALL_MATH_COMMON
#define UDX_WALL_MATH_COMMON
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
#endif /* UDX_WALL_MATH_COMMON */

#ifndef UDX_WALL_TFORM_TYPE_COMMON
#define UDX_WALL_TFORM_TYPE_COMMON
struct Tform_v { float o[3], s[3]; };
#endif
#ifndef UDX_WALL_TFORM_DEV_COMMON
#define UDX_WALL_TFORM_DEV_COMMON
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
#endif
#ifndef UDX_WALL_TEX3D_TYPE_COMMON
#define UDX_WALL_TEX3D_TYPE_COMMON
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
#endif
#ifndef UDX_WALL_SDF_TYPE_COMMON
#define UDX_WALL_SDF_TYPE_COMMON
/* view */
struct Sdf_v {
    Tex3d_v tex;
    Tform_v t;
    float cheap_threshold;
};
#endif /* UDX_WALL_SDF_TYPE_COMMON */
struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);

void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*); // <1>
void sdf_get_view(const Sdf*, /**/ Sdf_v*);                  // <2>

void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp); // <1>
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);          // <2>
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);                          // <3>

void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);

#ifndef UDX_WALL_SDF_DEV_COMMON
#define UDX_WALL_SDF_DEV_COMMON
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

_I_ float3 sdf_ugrad(const Sdf_v *texsdf, const float3 *r) // <1>
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

_I_ bool sdf_far(const Sdf_v *sdf, float x, float y, float z) // <2>
{
    return cheap_sdf(sdf, x, y, z) <= sdf->cheap_threshold - 1 ;
}

_I_ float sdf_eval(const Sdf_v *sdf0, float x, float y, float z)  // <3>
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
#endif /* UDX_WALL_SDF_DEV_COMMON */

struct Sdf;
struct WvelStep;
struct Coords;
struct Particle;

void bounce_back(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp);


namespace sdf_bb_dev {
enum {
    MAX_RESCUE = 8,
    MAX_NEWTON = 2
};

static __device__ void p2rv(const Particle *pp, int i, /**/ float3 *r, float3 *v) {
    enum {X, Y, Z};
    Particle p = pp[i];
    r->x = p.r[X];
    r->y = p.r[Y];
    r->z = p.r[Z];

    v->x = p.v[X];
    v->y = p.v[Y];
    v->z = p.v[Z];
}

static __device__ void rv2p(float3 r, float3 v, int i, /**/ Particle *pp) {
    enum {X, Y, Z};
    Particle p = {r.x, r.y, r.z, v.x, v.y, v.z};
    pp[i] = p;
}

static __device__ bool is_small(float f) {return fabs(f) < 1e-6f;}
static __device__ void crop(float dt, float *t) {
    if (*t < -dt) *t = -dt;
    if (*t >   0) *t = 0;
}

template <typename Wvel_v>
static __device__ void rescue(Wvel_v wv, Coords_v c, Sdf_v *texsdf, float currsdf, /* io */ float3 *r, float3 *v) {
    float sdf0, jump;
    float3 dsdf;
    int l;

    dsdf = sdf_ugrad(texsdf, r);
    sdf0 = currsdf;

    axpy(-sdf0, &dsdf, /**/ r);

    for (l = MAX_RESCUE; l >= 1; --l) {
        if (sdf_eval(texsdf, r->x, r->y, r->z) < 0) {
            bounce_vel(wv, c, *r, /**/ v);
            return;
        }
        jump = 1.1f * sdf0 / (1 << l);
        axpy(-jump, &dsdf, /**/ r);
    }
}

template <typename Wvel_v>
static __device__ void bounce_back_1p(float dt, Wvel_v wv, Coords_v c, Sdf_v *texsdf, float currsdf,
                                      /* io */ float3 *r, float3 *v) {
    float3 r0, rc, rw, dsdf;
    float phi, dphi, t;
    int l;

    r0 = *r;
    // get previous position
    axpy(-dt, v, /**/ &r0);

    if (sdf_eval(texsdf, r0.x, r0.y, r0.z) >= 0) {
        rescue(wv, c, texsdf, currsdf, /* io */ r, v);
        return;
    }

    /* use Newton iterations to solve sdf(rw) = 0 */
    /* where rw = r + t * v, t in [-dt, 0]        */
    t = 0;
    for (l = 0; l < MAX_NEWTON; ++l) {
        rc = *r;
        axpy(t, v, /**/ &rc);
        phi = sdf_eval(texsdf, rc.x, rc.y, rc.z);
        dsdf = grad(texsdf, &rc);
        dphi = dot<float> (v, &dsdf);

        if (is_small(dphi))
            break;

        t -= phi/dphi;
        crop(dt, &t);
    }

    rw = *r;
    axpy(t, v, /**/ &rw);
    bounce_vel(wv, c, rw, /**/ v);

    *r = rw;
    axpy(-t, v, /**/ r);

    if (sdf_eval(texsdf, r->x, r->y, r->z) >= 0)
        *r = r0;
}

template <typename Wvel_v>
__global__ void bounce_back(float dt, Wvel_v wv, Coords_v c, Sdf_v texsdf, int n, /**/ Particle *pp) {
    float s, currsdf;
    float3 r, v;
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    p2rv(pp, i, /**/ &r, &v);

    s = cheap_sdf(&texsdf, r.x, r.y, r.z);

    if (s >= texsdf.cheap_threshold) {
        currsdf = sdf_eval(&texsdf, r.x, r.y, r.z);
        if (currsdf >= 0) {
            bounce_back_1p(dt, wv, c, &texsdf, currsdf, /*io*/ &r, &v);
            rv2p(r, v, i, /**/ pp);
        }
    }
}
}
template <typename Wvel_v>
static void apply(float dt, Wvel_v wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    Sdf_v sdf_v;
    Coords_v coordsv;
    sdf_get_view(sdf, &sdf_v);
    coords_get_view(c, &coordsv);
    KL(sdf_bb_dev::bounce_back, (k_cnf(n)), (dt, wv, coordsv, sdf_v, n, /**/ pp));
}

void bounce_back(float dt, const WvelStep *w, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    switch (wvel_get_type(w)) {
    case WALL_VEL_V_CSTE: {
        WvelCste_v wv;
        wvel_get_view(w, /**/ &wv);
        apply(dt, wv, c, sdf, n, /**/ pp);
        break;
    }
    case WALL_VEL_V_SHEAR: {
        WvelShear_v wv;
        wvel_get_view(w, /**/ &wv);
        apply(dt, wv, c, sdf, n, /**/ pp);
        break;
    }
    case WALL_VEL_V_HS: {
        WvelHS_v wv;
        wvel_get_view(w, /**/ &wv);
        apply(dt, wv, c, sdf, n, /**/ pp);
        break;
    }
    };
}

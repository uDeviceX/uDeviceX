// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/wall/imp.cu */
#include <mpi.h>
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

/* begin src/conf/imp.h */
struct Config;
struct float3;
struct int3;

// tag::mem[]
void conf_ini(/**/ Config **c);
void conf_fin(/**/ Config *c);
// end::mem[]

// tag::ini[]
void conf_read(int argc, char **argv, /**/ Config *cfg);
// end::ini[]

// tag::lookup[]
void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::lookup[]

// tag::ns[]
void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);
// end::ns[]

// tag::opt[]
bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::opt[]

// tag::set[]
void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);
// end::set[]

// tag::dump[]
void conf_write_exe(const Config *cfg, FILE *stream);     // <1>
void conf_write_history(const Config *cfg, FILE *stream); // <2>
// end::dump[]
/* end src/conf/imp.h */

/* begin src/math/rnd/imp.h */
// tag::interface[]
struct RNDunif;                                             // <1>
void rnd_ini(int x, int y, int z, int c, /**/ RNDunif **r); // <2>
void rnd_fin(RNDunif *r);                                   // <3>
float rnd_get(RNDunif *r);                                  // <4>
// end::interface[]
/* end src/math/rnd/imp.h */

/* begin src/coords/type.h */
// tag::view[]
struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
// end::view[]
/* end src/coords/type.h */
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

/* begin src/math/tform/type.h */
struct Tform_v { float o[3], s[3]; };
/* end src/math/tform/type.h */
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
/* begin src/wall/sdf/imp.h */
struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

// tag::mem[]
void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);
// end::mem[]

// tag::int[]
void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*); // <1>
void sdf_get_view(const Sdf*, /**/ Sdf_v*);                  // <2>
// end::int[]

// tag::tools[]
void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp); // <1>
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);          // <2>
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);                          // <3>
// end::tools[]

// tag::bounce[]
void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);
// end::bounce[]
/* end src/wall/sdf/imp.h */

/* begin src/struct/partlist/type.h */
enum {
    ALIVE = 0,
    DEAD  = 1
};

/* structure of particle                           */
/* optional "deathlist" for particles to be killed */

// tag::type[]
struct PartList {
    const Particle *pp;   /* particles array          */
    const int *deathlist; /* what pp is dead or alive */
};
// end::type[]
/* end src/struct/partlist/type.h */
/* begin src/algo/scan/imp.h */
struct Scan;

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
// end::interface[]
/* end src/algo/scan/imp.h */
/* begin src/clist/imp.h */
struct Clist;
struct ClistMap;

// tag::mem[]
void clist_ini(int LX, int LY, int LZ, /**/ Clist **c);
void clist_fin(/**/ Clist *c);

void clist_ini_map(int maxp, int nA, const Clist *c, /**/ ClistMap **m);
void clist_fin_map(ClistMap *m);
// end::mem[]

// tag::access[]
const int* clists_get_ss(const Clist *c);
const int* clists_get_cc(const Clist *c);
int3       clists_get_dim(const Clist *c);
int        clists_get_n(const Clist *c);
const uint* clist_get_ids(const ClistMap *m);
// end::access[]

// tag::int[]
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
// end::int[]
/* end src/clist/imp.h */

/* begin src/io/restart/imp.h */
struct Coords;
struct Particle;
struct Rigid;

// tag::id[]
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
// end::id[]

// tag::stream[]
typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);
// end::stream[]

// tag::write[]
void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp); // <1>
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);      // <2>
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);    // <3>
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter); // <4>
 // end::write[]

// tag::read[]
void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);
// end::read[]
/* end src/io/restart/imp.h */

/* begin src/wall/exch/imp.h */
struct int3;
struct Particle;
// tag::int[]
void wall_exch_pp(MPI_Comm cart, int numdensity, int3 L, int maxn, /*io*/ Particle *pp, int *n);
// end::int[]
/* end src/wall/exch/imp.h */
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

/* begin src/wall/imp.h */
struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct Coords;
struct Sdf;
struct int3;
struct float4;
struct Config;

// tag::struct[]
struct WallQuants {
    float4 *pp; /* particle positions xyzo xyzo ... */
    int n;      /* number of particles              */
    int3 L;     /* subdomain size                   */
};

struct WallTicket;
struct WallRepulsePrm;

// end::struct[]

// tag::mem[]
void wall_ini_quants(int3 L, WallQuants*);
void wall_ini_ticket(int3 L, WallTicket**);

void wall_fin_quants(WallQuants*);
void wall_fin_ticket(WallTicket*);
// end::mem[]

// tag::gen[]
void wall_gen_quants(MPI_Comm, int numdensity, int maxn, const Sdf*, /* io */ int *o_n, Particle *o_pp, /**/ WallQuants*); // <1>
void wall_gen_ticket(const WallQuants*, WallTicket*); // <2>
// end::gen[]

// tag::start[]
void wall_strt_quants(MPI_Comm, const char *base, int maxn, WallQuants*); // <1>
void wall_strt_dump_templ(MPI_Comm, const char *base, const WallQuants*); // <2>
// end::start[]

// tag::int[]
void wall_force(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                const WallTicket*, int n, const PaArray*, const FoArray*); // <1>

void wall_force_adhesion(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                         const WallTicket*, int n, const PaArray*, const FoArray*); // <2>

void wall_repulse(const Sdf*, const WallRepulsePrm*, long n, const PaArray*, const FoArray*); // <3>
// end::int[]

// tag::repulse[]
void wall_repulse_prm_ini(float lambda, WallRepulsePrm**); // <1>
void wall_repulse_prm_ini_conf(const Config*, const char *ns, WallRepulsePrm**); // <2>
void wall_repulse_prm_fin(WallRepulsePrm*); // <3>
// end::repulse[]
/* end src/wall/imp.h */

namespace wall_dev {
/* begin src/wall/dev/main.h */
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
/* end src/wall/dev/main.h */
}

/* begin src/wall/imp/type.h */
struct WallTicket {
    RNDunif *rnd;        /* rng on host                                        */
    Clist *cells;        /* cell lists (always the same, no need to store map) */
    Texo<int> texstart;  /* texture of starts from clist                       */
    Texo<float4> texpp;  /* texture of particle positions                      */
};
/* end src/wall/imp/type.h */
/* begin src/wall/imp/ini.h */
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
/* end src/wall/imp/ini.h */
/* begin src/wall/imp/fin.h */
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
/* end src/wall/imp/fin.h */
/* begin src/wall/imp/strt.h */
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
/* end src/wall/imp/strt.h */
/* begin src/wall/imp/generate.h */
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
/* end src/wall/imp/generate.h */
/* begin src/wall/imp/force.h */
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
/* end src/wall/imp/force.h */
/* begin src/wall/imp/repulse.h */
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
/* end src/wall/imp/repulse.h */
/* end src/wall/imp.cu */

#line 1 "src/wall/sdf/array3d/imp.cu"

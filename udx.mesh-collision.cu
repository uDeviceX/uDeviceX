// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/mesh/collision/imp.cu */
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

#ifndef UDX_MESH_DEF_COMMON
#define UDX_MESH_DEF_COMMON
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
#endif
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
#ifndef UDX_MESH_DEVICE_API_COMMON
#define UDX_MESH_DEVICE_API_COMMON
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

#ifndef UDX_MESH_COMMON_CUH
#define UDX_MESH_COMMON_CUH
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
#ifndef UDX_MESH_TRIANGLES_TYPE
#define UDX_MESH_TRIANGLES_TYPE
/* begin src/mesh/triangles/type.h */
// tag::interface[]
struct Triangles {
    int4 *tt;
    int nt;
};
// end::interface[]
/* end src/mesh/triangles/type.h */
#endif
/* begin src/mesh/triangles/imp.h */
struct Triangles;
struct MeshRead;

// tag::interface[]
void triangles_ini(MeshRead*, /**/ Triangles**);
void triangles_fin(Triangles*);
// end::interface[]
/* end src/mesh/triangles/imp.h */

#ifndef UDX_MESH_KL_COMMON
#define UDX_MESH_KL_COMMON
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
#endif
/* begin src/rigid/imp.h */
struct RigPinInfo;
struct int3;
struct Config;

// tag::enum[]
enum {NOT_PERIODIC = -1};
// end::enum[]

// tag::pin[]
void rig_pininfo_ini(RigPinInfo **); // <1>
void rig_pininfo_fin(RigPinInfo *); // <2>
void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *); // <3>
void rig_pininfo_set_pdir(int pdir, RigPinInfo *); // <4>
void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *); // <5>

int rig_pininfo_get_pdir(const RigPinInfo *); // <6>
// end::pin[]

struct Particle;
struct Rigid;
struct Force;

// tag::upd[]
void rig_reinit_ft(const int ns, /**/ Rigid *ss); // <1>
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns, /**/ Particle *pp, Rigid *ss); // <2>
void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0, /**/ Particle *pp); // <3>
void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv, /**/ Particle *pp); // <4>
// end::upd[]

/* end src/rigid/imp.h */

/* begin src/mesh/collision/imp.h */
struct float3;
struct Particle;
struct int4;
struct Triangles;

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt);

void collision_label_ini(long n, int lab_out, /**/ int *labels);
void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri,
                     int nv, int nm, const Particle *i_pp,
                     const float3 *minext, const float3 *maxext,
                     int lab_in, /**/ int *labels);
/* end src/mesh/collision/imp.h */

enum {X, Y, Z};

static __host__ __device__ bool same_side(const float *x, const float *p, const float *a, const float *b, const float *inplane) {
    const float n[3] = {a[Y] * b[Z] - a[Z] * b[Y],
                        a[Z] * b[X] - a[X] * b[Z],
                        a[X] * b[Y] - a[Y] * b[X]};

    const float ndx = n[X] * (x[X] - inplane[X]) + n[Y] * (x[Y] - inplane[Y]) + n[Z] * (x[Z] - inplane[Z]);
    const float ndp = n[X] * (p[X] - inplane[X]) + n[Y] * (p[Y] - inplane[Y]) + n[Z] * (p[Z] - inplane[Z]);

    return ndx * ndp > 0;
}

static __host__ __device__ bool in_tetrahedron(const float *x, const float *A, const float *B, const float *C, const float *D) {
    const float AB[3] = {B[X] - A[X], B[Y] - A[Y], B[Z] - A[Z]};
    const float AC[3] = {C[X] - A[X], C[Y] - A[Y], C[Z] - A[Z]};
    const float AD[3] = {D[X] - A[X], D[Y] - A[Y], D[Z] - A[Z]};

    const float BC[3] = {C[X] - B[X], C[Y] - B[Y], C[Z] - B[Z]};
    const float BD[3] = {D[X] - B[X], D[Y] - B[Y], D[Z] - B[Z]};

    return
        same_side(x, A, BC, BD, D) &&
        same_side(x, B, AD, AC, D) &&
        same_side(x, C, AB, BD, D) &&
        same_side(x, D, AB, AC, A);
}

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt) {
    int c = 0;
    float origin[3] = {0, 0, 0};

    if (spdir != NOT_PERIODIC)
        origin[spdir] = r[spdir];

    for (int i = 0; i < nt; ++i) {
        int4 t = tt[i];
        if (in_tetrahedron(r, vv + 3*t.x, vv + 3*t.y, vv + 3*t.z, origin)) ++c;
    }
    return c%2;
}

namespace collision_dev
{
__global__ void init_tags(const int n, const int color, /**/ int *tags) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < n) tags[gid] = color;
}

union Pos {
    float2 f2[2];
    struct { float r[3]; float dummy; };
};

static __device__ Pos fetchPos(const Particle *vert, const int id) {
    const float2 *pp = (const float2*) vert;
    Pos r;
    r.f2[0] = pp[3 * id + 0];
    r.f2[1] = pp[3 * id + 1];
    return r;
}

static __device__ bool inside_box(const float r[3], float3 lo, float3 hi) {
    enum {X, Y, Z};
    return
        r[X] >= lo.x && r[X] <= hi.x &&
        r[Y] >= lo.y && r[Y] <= hi.y &&
        r[Z] >= lo.z && r[Z] <= hi.z;
}

/* assume nm blocks along y */
__global__ void label(int pdir, const Particle *pp, const int n, const Particle *vert, const int nv,
                      Triangles tri, const float3 *minext, const float3 *maxext,
                      int lab_in, /**/ int *labels) {
    int i, sid, gid, count, mbase;
    Particle p;
    Pos a, b, c;
    float3 lo, hi;
    int4 t;
    sid = blockIdx.y;
    gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= n) return;

    count = 0;

    p = pp[gid];

    lo = minext[sid];
    hi = maxext[sid];
    if (!inside_box(p.r, lo, hi)) return;

    float origin[3] = {0, 0, 0};
    if (pdir != NOT_PERIODIC) origin[pdir] = p.r[pdir];

    mbase = nv * sid;
    for (i = 0; i < tri.nt; ++i) {
        t = tri.tt[i];

        a = fetchPos(vert, mbase + t.x);
        b = fetchPos(vert, mbase + t.y);
        c = fetchPos(vert, mbase + t.z);

        if (in_tetrahedron(p.r, a.r, b.r, c.r, origin)) ++count;
    }

    // dont consider the case of inside several solids
    if (count % 2) atomicExch(labels + gid, lab_in);
}
}

void collision_label_ini(long n, int lab_out, /**/ int *labels) {
    KL(collision_dev::init_tags, (k_cnf(n)), (n, lab_out, /**/ labels));
}

void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri,
                     int nv, int nm, const Particle *i_pp,
                     const float3 *minext, const float3 *maxext,
                     int lab_in, /**/ int *labels) {
    enum {THR = 128};

    dim3 thrd(THR, 1);
    dim3 blck(ceiln(n, THR), nm);

    if (nm == 0 || n == 0) return;

    KL(collision_dev::label, (blck, thrd),
       (pdir, pp, n, i_pp, nv, *tri, minext, maxext, lab_in, /**/ labels));

}
/* end src/mesh/collision/imp.cu */

#line 1 "src/mesh/cylindrical/imp.cpp"

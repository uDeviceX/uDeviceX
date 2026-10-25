// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rigid/conf.cpp */
#include <stdio.h>
#include <vector_types.h>

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

void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *pi) {
    int3 com, axis;
    int pdir;

    UC(conf_lookup_int3_ns(cfg, ns, "pin_com", &com));
    UC(conf_lookup_int3_ns(cfg, ns, "pin_axis", &axis));
    UC(conf_lookup_int_ns(cfg, ns, "pdir", &pdir));

    UC(rig_pininfo_set(com, axis, pi));
    UC(rig_pininfo_set_pdir(pdir, pi));
}
/* end src/rigid/conf.cpp */

#line 1 "src/rigid/imp.cu"


/* begin src/rigid/imp.cu */
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

/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
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



/* begin src/rigid/imp/type.h */
struct RigPinInfo {
    int3 com;  /* what components of com stay fixed */
    int3 axis;  /* what components of rotation stay fixed */
    int pdir; /* direction of periodicity */
};
/* end src/rigid/imp/type.h */

namespace rigid_dev {
/* begin src/rigid/dev/utils.h */
static __device__ float dot(const float *v, const float *u) {
    enum {X, Y, Z};
    return v[X]*u[X] + v[Y]*u[Y] + v[Z]*u[Z];
}

static __device__ void reject(/**/ float *v, float *u) {
    enum {X, Y, Z};
    const float d = dot(v, u);
    v[X] -= d*u[X]; v[Y] -= d*u[Y]; v[Z] -= d*u[Z];
}

static __device__ void normalize(/**/ float *v) {
    enum {X, Y, Z};
    const float s = rsqrtf(dot(v, v));
    v[X] *= s; v[Y] *= s; v[Z] *= s;
}

__device__ void gram_schmidt(/**/ float *e0, float *e1, float *e2) {
    /* :TODO: use better more stable version of Gram-Schmidt */
    normalize(e0);

    reject(e1, e0);
    normalize(e1);

    reject(e2, e0);
    reject(e2, e1);
    normalize(e2);
}

__device__ void rot_e(float dt, const float *om, /**/ float *e) {
    enum {X, Y, Z};
    const float omx = om[X], omy = om[Y], omz = om[Z];
    const float ex = e[X], ey = e[Y], ez = e[Z];

    const float vx = omy*ez - omz*ey;
    const float vy = omz*ex - omx*ez;
    const float vz = omx*ey - omy*ex;

    e[X] += vx*dt; e[Y] += vy*dt; e[Z] += vz*dt;
}
/* end src/rigid/dev/utils.h */
/* begin src/rigid/dev/main.h */
__global__ void rot_referential(float dt, const int ns, Rigid *ss) {
    int i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < ns) {
        Rigid s = ss[i];
        rot_e(dt, s.om, /**/ s.e0);
        rot_e(dt, s.om, /**/ s.e1);
        rot_e(dt, s.om, /**/ s.e2);

        gram_schmidt(/**/ s.e0, s.e1, s.e2);
        ss[i] = s;
    }
}

static __device__ void atomicAdd3(float d[3], float3 s) {
    enum {X, Y, Z};
    atomicAdd(d + X, s.x);
    atomicAdd(d + Y, s.y);
    atomicAdd(d + Z, s.z);
}

static __device__ float3 fetch_force(int i, const Force *ff) {
    enum {X, Y, Z};
    float3 f;
    f.x = ff[i].f[X];
    f.y = ff[i].f[Y];
    f.z = ff[i].f[Z];
    return f;
}

__global__ void add_f_to(const int nps, const Particle *pp, const Force *ff, /**/ Rigid *ss) {
    enum {X, Y, Z};
    int gid, sid, i;
    gid = blockIdx.x * blockDim.x + threadIdx.x;
    sid = blockIdx.y;

    i = sid * nps + gid;

    float3 to, f = make_float3(0, 0, 0);
    Particle   p = {0, 0, 0, 0, 0, 0};

    if (gid < nps) {
        f = fetch_force(i, ff);
        p = pp[i];
    }

    const float dr[3] = {p.r[X] - ss[sid].com[X],
                         p.r[Y] - ss[sid].com[Y],
                         p.r[Z] - ss[sid].com[Z]};

    to.x = dr[Y] * f.z - dr[Z] * f.y;
    to.y = dr[Z] * f.x - dr[X] * f.z;
    to.z = dr[X] * f.y - dr[Y] * f.x;

    f =  warpReduceSum(f);
    to = warpReduceSum(to);

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd3(ss[sid].fo, f);
        atomicAdd3(ss[sid].to, to);
    }
}

__global__ void reinit_ft(const int ns, Rigid *ss) {
    enum {X, Y, Z};
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < ns) {
        Rigid *s = ss + gid;
        s->fo[X] = s->fo[Y] = s->fo[Z] = 0.f;
        s->to[X] = s->to[Y] = s->to[Z] = 0.f;
    }
}

__global__ void update_om_v(RigPinInfo pi, float dt, const int ns, Rigid *ss) {
    enum {X, Y, Z};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < ns) {
        Rigid s = ss[gid];
        const float *A = s.Iinv, *b = s.to;
        const float sc = dt/s.mass;
        const float dom[3] = {A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z],
                              A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z],
                              A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z]};

        s.om[X] += dom[X]*dt;
        s.om[Y] += dom[Y]*dt;
        s.om[Z] += dom[Z]*dt;

        if (pi.axis.x) s.om[X] = 0;
        if (pi.axis.y) s.om[Y] = 0;
        if (pi.axis.z) s.om[Z] = 0;


        s.v[X] += s.fo[X] * sc;
        s.v[Y] += s.fo[Y] * sc;
        s.v[Z] += s.fo[Z] * sc;

        if (pi.com.x) s.v[X] = 0;
        if (pi.com.y) s.v[Y] = 0;
        if (pi.com.z) s.v[Z] = 0;

        ss[threadIdx.x] = s;
    }
}

__global__ void update_com(float dt, int ns, Rigid *ss) {
    int sid, c, i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    sid = i / 3;
    c   = i % 3;

    if (sid < ns)
        ss[sid].com[c] += ss[sid].v[c]*dt;
}

__global__ void update_pp(const int nps, const float *rr0, const Rigid *ss, /**/ Particle *pp) {
    enum {X, Y, Z};
    int pid, sid, i;
    Rigid s;
    float x, y, z, dx, dy, dz, omx, omy, omz;
    Particle p;
    const float *r0;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    sid = blockIdx.y;
    i = sid * nps + pid;

    s = ss[sid];
    omx = s.om[X]; omy = s.om[Y]; omz = s.om[Z];

    if (pid < nps) {
        p = pp[i];
        r0 = &rr0[3*pid];
        x = r0[X]; y = r0[Y]; z = r0[Z];

        dx = x * s.e0[X] + y * s.e1[X] + z * s.e2[X];
        dy = x * s.e0[Y] + y * s.e1[Y] + z * s.e2[Y];
        dz = x * s.e0[Z] + y * s.e1[Z] + z * s.e2[Z];

        p.v[X] = s.v[X] + omy * dz - omz * dy;
        p.v[Y] = s.v[Y] + omz * dx - omx * dz;
        p.v[Z] = s.v[Z] + omx * dy - omy * dx;

        p.r[X] = s.com[X] + dx;
        p.r[Y] = s.com[Y] + dy;
        p.r[Z] = s.com[Z] + dz;

        pp[i] = p;
    }
}

__global__ void update_mesh(float dt, const Rigid *ss_dev, const int nv, const float *vv, /**/ Particle *pp) {
    enum {X, Y, Z};
    const int sid = blockIdx.y; // solid Id
    const Rigid *s = ss_dev + sid;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int vid = sid * nv + i;

    if (i < nv) {
        const float x = vv[3*i + X];
        const float y = vv[3*i + Y];
        const float z = vv[3*i + Z];

        const Particle p0 = pp[vid];
        Particle p;

        p.r[X] = x * s->e0[X] + y * s->e1[X] + z * s->e2[X] + s->com[X];
        p.r[Y] = x * s->e0[Y] + y * s->e1[Y] + z * s->e2[Y] + s->com[Y];
        p.r[Z] = x * s->e0[Z] + y * s->e1[Z] + z * s->e2[Z] + s->com[Z];

        p.v[X] = (p.r[X] - p0.r[X]) / dt;
        p.v[Y] = (p.r[Y] - p0.r[Y]) / dt;
        p.v[Z] = (p.r[Z] - p0.r[Z]) / dt;

        pp[vid] = p;
    }
}
/* end src/rigid/dev/main.h */
}

/* begin src/rigid/imp/main.h */
enum {NTHRD=128};

void rig_pininfo_ini(RigPinInfo **pi) {
    RigPinInfo *p;
    EMALLOC(1, pi);
    p = *pi;
    p->pdir = NOT_PERIODIC;
}

void rig_pininfo_fin(RigPinInfo *pi) {
    EFREE(pi);
}

void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *pi) {
    pi->com  = com;
    pi->axis = axis;
}

void rig_pininfo_set_pdir(int pdir, RigPinInfo *pi) {
    pi->pdir = pdir;
}

int rig_pininfo_get_pdir(const RigPinInfo *pi) {
    return pi->pdir;
}

void rig_reinit_ft(const int nsolid, /**/ Rigid *ss) {
    KL(rigid_dev::reinit_ft, (k_cnf(nsolid)), (nsolid, /**/ ss));
}

static bool pinned_com(const RigPinInfo *pi) {
    int3 com = pi->com;
    return com.x && com.y && com.z;
}

void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns, /**/ Particle *pp, Rigid *ss) {
    if (ns < 1) return;

    const int nps = n / ns; /* number of particles per solid */

    const dim3 nblck ( ceiln(nps, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );

    KL(rigid_dev::add_f_to, ( nblck, nthrd ), (nps, pp, ff, /**/ ss));
    KL(rigid_dev::update_om_v, (1, ns), (*pi, dt, ns, /**/ ss));
    if (!pinned_com(pi))
        KL(rigid_dev::update_com, (1, 3*ns ), (dt, ns, /**/ ss));
    KL(rigid_dev::rot_referential, (1, ns), (dt, ns, /**/ ss));

    KL(rigid_dev::update_pp, ( nblck, nthrd ), (nps, rr0, ss, /**/ pp));
}

void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0, /**/ Particle *pp) {
    if (ns < 1) return;

    const dim3 nblck ( ceiln(nps, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );

    KL(rigid_dev::update_pp, ( nblck, nthrd ), (nps, rr0, ss, /**/ pp));
}

void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv, /**/ Particle *pp) {
    const dim3 nblck ( ceiln(nv, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );
    KL(rigid_dev::update_mesh, (nblck, nthrd ), (dt, ss, nv, vv, /**/ pp));
}
/* end src/rigid/imp/main.h */
/* end src/rigid/imp.cu */

#line 1 "src/scheme/force/conf.cpp"

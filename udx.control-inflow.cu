// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/control/inflow/conf.cpp */
#include <stdlib.h>
#include <stdio.h>
#include <vector_types.h>

#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
/* begin src/coords/type.h */
// tag::view[]
struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
// end::view[]
/* end src/coords/type.h */
#endif /* UDX_CONTROL_COORDS_COMMON */
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
#ifndef UDX_CONTROL_UTILS_COMMON
#define UDX_CONTROL_UTILS_COMMON
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
#endif /* UDX_CONTROL_UTILS_COMMON */
#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON
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
#endif /* UDX_CONTROL_ERROR_COMMON */

/* begin src/control/inflow/imp.h */
struct Inflow;
struct Config;
struct Particle;
struct int2;
struct float3;
struct Coords;

// tag::mem[]
void inflow_ini(int2 nc, Inflow **i);
void inflow_fin(Inflow *i);
// end::mem[]

// tag::ini[]
void inflow_ini_velocity(Inflow *i);

void inflow_ini_params_plate(const Coords *c, float3 o, int dir, float L1, float L2,
                             float3 u, bool upoiseuille, bool vpoiseuille,
                             /**/ Inflow *i);
void inflow_ini_params_circle(const Coords *c, float3 o, float R, float H, float u, bool poiseuille,
                              /**/ Inflow *i);
// end::ini[]

// tag::cnf[]
void inflow_ini_params_conf(const Coords *coords, const Config *cfg, /**/ Inflow *i);
// end::cnf[]

// tag::int[]
void inflow_create_pp(float kBT, int numdensity, float dt, Inflow *i, int *n, Particle *pp); // <1>
void inflow_create_pp_cc(float kBT, int numdensity, float dt, int newcolor, Inflow *i, int *n, Particle *pp, int *cc); // <2>
// end::int[]
/* end src/control/inflow/imp.h */

void inflow_ini_params_conf(const Coords *coords, const Config *cfg, /**/ Inflow *i) {
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));

    if      (same_str(type, "circle")) {
        float R, H, U;
        int poiseuille;
        float3 center;

        UC(conf_lookup_float(cfg, "inflow.R", &R));
        UC(conf_lookup_float(cfg, "inflow.H", &H));
        UC(conf_lookup_float(cfg, "inflow.U", &U));
        UC(conf_lookup_bool(cfg, "inflow.poiseuille", &poiseuille));
        UC(conf_lookup_float3(cfg, "inflow.center", &center));
        UC(inflow_ini_params_circle(coords, center, R, H, U, poiseuille, /**/ i));
    }
    else if (same_str(type, "plate")) {
        int upois, vpois, dir;
        float3 origin, u;
        float L1, L2;

        UC(conf_lookup_float(cfg, "inflow.L1", &L1));
        UC(conf_lookup_float(cfg, "inflow.L2", &L2));
        UC(conf_lookup_int(cfg, "inflow.direction", &dir));
        UC(conf_lookup_bool(cfg, "inflow.upoiseuille", &upois));
        UC(conf_lookup_bool(cfg, "inflow.vpoiseuille", &vpois));
        UC(conf_lookup_float3(cfg, "inflow.origin", &origin));
        UC(conf_lookup_float3(cfg, "inflow.u", &u));

        UC(inflow_ini_params_plate(coords, origin, dir, L1, L2, u, upois, vpois, /**/ i));
    }
    else {
        ERR("unknown inflow type <%s>", type);
    }
}

/* end src/control/inflow/conf.cpp */

#line 1 "src/control/inflow/imp.cu"


/* begin src/control/inflow/imp.cu */
#include <stdio.h>
#include <curand_kernel.h>

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


#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON
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
#endif /* UDX_CONTROL_ERROR_COMMON */

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
#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
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
#endif /* UDX_CONTROL_KL_COMMON */
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
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
#endif /* UDX_CONTROL_D_API_COMMON */

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
#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
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
#endif /* UDX_CONTROL_TYPE_COMMON */


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
#ifndef UDX_CONTROL_MATH_COMMON
#define UDX_CONTROL_MATH_COMMON
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
#endif /* UDX_CONTROL_MATH_COMMON */



/* begin src/control/inflow/plate/type.h */
struct ParamsPlate {
    float3 o, a, b;
};

struct VParamsPlate {
    float3 u;
    bool upoiseuille, vpoiseuille;
};
/* end src/control/inflow/plate/type.h */
/* begin src/control/inflow/plate/dev.h */
static __device__ void coords2pos(ParamsPlate p, float2 xi, /**/ float3 *r) {
    *r = p.o;
    axpy(xi.x, &p.a, /**/ r);
    axpy(xi.y, &p.b, /**/ r);
}

static __device__ float3 get_normal(ParamsPlate p, int2 nc, int i, int j) {
    float3 n;
    cross(&p.a, &p.b, /**/ &n);
    scal(1.f / (nc.x * nc.y), /**/ &n);
    return n;
}

static __device__ void coords2vel(VParamsPlate vp, ParamsPlate p, float2 xi, /**/ float3 *u) {
    float fact;
    fact = 1.f;
    if (vp.upoiseuille)
        fact *= 4 * xi.x * (1 - xi.x);

    if (vp.vpoiseuille)
        fact *= 4 * xi.y * (1 - xi.y);

    *u = vp.u;
    scal(fact, /**/ u);
}
/* end src/control/inflow/plate/dev.h */

/* begin src/control/inflow/circle/type.h */
struct ParamsCircle {
    float3 o;   /* center of cylinder */
    float R, H; /* radius, hight      */
};

struct VParamsCircle {
    float u;
    bool poiseuille;
};
/* end src/control/inflow/circle/type.h */
/* begin src/control/inflow/circle/dev.h */
static __device__ void coords2pos(ParamsCircle p, float2 xi, /**/ float3 *r) {
    float th, h, cth, sth;
    th = xi.x * 2 * M_PI;
    h = xi.y * p.H;
    cth = cos(th);
    sth = sin(th);

    *r = p.o;
    r->x += p.R * cth;
    r->y += p.R * sth;
    r->z += h;
}

static __device__ float3 get_normal(ParamsCircle p, int2 nc, int i, int j) {
    float3 n;
    float th, cth, sth, area;
    th = (i + 0.5f) / (nc.x) * 2 * M_PI;
    cth = cos(th);
    sth = sin(th);

    n = make_float3(cth, sth, 0);
    area = p.H * p.R * 2 * M_PI / (nc.x * nc.y);
    scal(area, /**/ &n);
    return n;
}

static __device__ void coords2vel(VParamsCircle vp, ParamsCircle p, float2 xi, /**/ float3 *u) {
    float th, cth, sth, fact;
    th = xi.x * 2 * M_PI;
    cth = cos(th);
    sth = sin(th);

    fact = 1.f;
    if (vp.poiseuille) {
        float y = xi.y;
        fact *= 4 * y * (1 - y);
    }

    *u = make_float3(cth, sth, 0);
    scal(fact * vp.u, /**/ u);
}
/* end src/control/inflow/circle/dev.h */


/* begin src/control/inflow/imp/type.h */
enum Type {
    TYPE_NONE,
    TYPE_PLATE,
    TYPE_CIRCLE,
};

struct Desc {
    curandState_t *rnds; /* random states on device         */
    float3 *uu;          /* target flux                     */
    float *cumflux;      /* cumulative flux                 */
    int *ndev;           /* number of particles on device   */
    int2 nc;             /* number of cells in 2 directions */
};

union ParamsU {
    ParamsPlate plate;
    ParamsCircle circle;
};

union VParamsU {
    VParamsPlate plate;
    VParamsCircle circle;
};

struct Inflow {
    Desc d;
    Type t;
    ParamsU p;
    VParamsU vp;
};

struct SolventWrap {
    Particle *pp;       /* particles               */
    int *cc;            /* colors                  */
    bool multisolvent;  /* true if color activated */
    int color;          /* color of new particles  */
};
/* end src/control/inflow/imp/type.h */

namespace inflow_dev {
/* begin src/control/inflow/dev/ini.h */
__global__ void ini_rnd(long seed, int n, curandState_t *rr) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &rr[i]);
}

__global__ void ini_flux(int n, /**/ curandState_t *rr, float *cumflux) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    cumflux[i] = curand_uniform(rr + i);
}
/* end src/control/inflow/dev/ini.h */
/* begin src/control/inflow/dev/main.h */
template <typename Par, typename VPar>
__global__ void ini_vel(VPar vparams, Par params, int2 nc, /**/ float3 *uu) {
    int i, xcid, ycid;
    float3 u;
    float2 xi;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nc.x * nc.y) return;

    xcid = i % nc.x;
    ycid = i / nc.x;

    xi.x = (xcid + 0.5f) / nc.x;
    xi.y = (ycid + 0.5f) / nc.y;

    coords2vel(vparams, params, xi, /**/ &u);

    uu[i] = u;
}

template <typename Par>
__global__ void cumulative_flux(float nxdt, Par params, int2 nc, const float3 *uu, /**/ float *cumflux) {
    int i, xcid, ycid;
    float dn;
    float3 normal, u;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nc.x * nc.y) return;

    xcid = i % nc.x;
    ycid = i / nc.x;

    // normal is scaled by area of element
    normal = get_normal(params, nc, xcid, ycid);
    u      = uu[i];
    dn = nxdt * dot<float>(&normal, &u);

    cumflux[i] += dn;
}

template <typename Par>
static __device__ Particle create_particle(float kBT, Par params, int2 nc,
                                           int xcid, int ycid, float3 u, curandState_t *rg) {
    float2 xi;
    float3 r;
    float sigma;
    xi.x = (xcid + curand_uniform(rg)) / nc.x;
    xi.y = (ycid + curand_uniform(rg)) / nc.y;

    coords2pos(params, xi, /**/ &r);

    sigma = sqrtf(kBT);
    u.x += curand_normal(rg) * sigma;
    u.y += curand_normal(rg) * sigma;
    u.z += curand_normal(rg) * sigma;

    // printf("create one particle: %g %g %g   %g %g %g\n", r.x, r.y, r.z, u.x, u.y, u.z);

    return Particle({r.x, r.y, r.z, u.x, u.y, u.z});
}

template <typename Par>
__global__ void create_particles(float kBT, Par params, int2 nc, const float3 *flux,
                                 /*io*/ curandState_t *rnds, float *cumflux, /**/ int *n, SolventWrap wrap) {
    int i, xcid, ycid, j, nnew, strt;
    float c;
    float3 f;
    curandState_t rndstate;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    xcid = i % nc.x;
    ycid = i / nc.x;

    if (i >= nc.x * nc.y) return;

    c = cumflux[i];
    nnew = int(c);

    if (nnew == 0) return;

    c -= nnew;
    cumflux[i] = c;

    f        = flux[i];
    rndstate = rnds[i];

    strt = atomicAdd(n, nnew);

    for (j = strt; j < strt + nnew; ++j) {
        wrap.pp[j] = create_particle(kBT, params, nc, xcid, ycid, f, /*io*/ &rndstate);
        if (wrap.multisolvent)
            wrap.cc[j] = wrap.color;
    }

    rnds[i] = rndstate;
}
/* end src/control/inflow/dev/main.h */
}

/* begin src/control/inflow/plate/imp.h */
static float crop(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float3 crop(float3 r, float3 lo, float3 hi) {
    r.x = crop(r.x, lo.x, hi.x);
    r.y = crop(r.y, lo.y, hi.y);
    r.z = crop(r.z, lo.z, hi.z);
    return r;
}

void inflow_ini_params_plate(const Coords *c, float3 o, int dir, float L1, float L2,
                             float3 u, bool upoiseuille, bool vpoiseuille,
                             /**/ Inflow *i) {
    enum {X, Y, Z};
    ParamsPlate *pp;
    VParamsPlate *vpp;
    float3 a, b, lo, hi;
    float3 co, ca, cb; // corners of the local plane : co (origin), ca, cb

    lo = make_float3(-xs(c)/2, -ys(c)/2, -zs(c)/2);
    hi = make_float3( xs(c)/2,  ys(c)/2,  zs(c)/2);

    pp = &i->p.plate;
    vpp = &i->vp.plate;

    a = b = make_float3(0, 0, 0);

    switch(dir) {
    case X:
        a.y = L1;
        b.z = L2;
        break;
    case Y:
        a.x = L1;
        b.z = L2;
        break;
    case Z:
        a.x = L1;
        b.y = L2;
        break;
    default:
        ERR("wrong direction");
        break;
    };

    // shift to local coordinates
    global2local(c, o, /**/ &o);

    co = ca = cb = o;
    add(&a, /**/ &ca);
    add(&b, /**/ &cb);

    co = crop(co, lo, hi);
    ca = crop(ca, lo, hi);
    cb = crop(cb, lo, hi);

    pp->o = co;
    diff(&ca, &co, /**/ &pp->a);
    diff(&cb, &co, /**/ &pp->b);

    vpp->u = u;
    vpp->upoiseuille = upoiseuille;
    vpp->vpoiseuille = vpoiseuille;

    i->t = TYPE_PLATE;
}
/* end src/control/inflow/plate/imp.h */
/* begin src/control/inflow/circle/imp.h */
static bool is_inside_subdomain(const Coords *c, float3 r) {
    bool inside = true;
    inside &= (xlo(c) <= r.x) && (r.x < xhi(c));
    inside &= (ylo(c) <= r.y) && (r.y < yhi(c));
    inside &= (zlo(c) <= r.z) && (r.z < zhi(c));
    return inside;
}

void inflow_ini_params_circle(const Coords *c, float3 o, float R, float H, float u, bool poiseuille,
                              /**/ Inflow *i) {
    enum {X, Y, Z};
    ParamsCircle *pp;
    VParamsCircle *vpp;

    if (is_inside_subdomain(c, o)) {
        pp  = &i->p.circle;
        vpp = &i->vp.circle;

        global2local(c, o, /**/ &o);

        pp->o = o;
        pp->R = R;
        pp->H = H;

        if (R > xs(c)/2 || R > ys(c)/2 || H > zs(c))
            ERR("dimensions should be smaller");

        vpp->u = u;
        vpp->poiseuille = poiseuille;

        i->t = TYPE_CIRCLE;
    } else {
        i->t = TYPE_NONE;
    }
}
/* end src/control/inflow/circle/imp.h */

/* begin src/control/inflow/imp/main.h */
static void ini_rnd(int n, curandState_t *rr) {
    long seed = 1234567;
    KL(inflow_dev::ini_rnd, (k_cnf(n)), (seed, n, rr));
}

static void ini_flux(int n, curandState_t *rr, float *cumflux) {
    KL(inflow_dev::ini_flux, (k_cnf(n)), (n, rr, cumflux));
}

void inflow_ini(int2 nc, Inflow **i) {
    int n;
    Inflow *ip;
    Desc *d;

    EMALLOC(1, i);

    ip = *i;
    d = &ip->d;

    n = nc.x * nc.y;
    d->nc = nc;

    Dalloc(&d->rnds,    n);
    Dalloc(&d->uu,      n);
    Dalloc(&d->cumflux, n);
    Dalloc(&d->ndev,    1);

    UC(ini_rnd(n, d->rnds));
    UC(ini_flux(n, d->rnds, d->cumflux));

    ip->t = TYPE_NONE;
}

static void ini_velocity(Type t, int2 nc, const ParamsU *p, const VParamsU *vp, /**/ float3 *uu) {
    int n = nc.x * nc.y;
    switch(t) {
    case TYPE_PLATE:
        KL(inflow_dev::ini_vel, (k_cnf(n)), (vp->plate, p->plate, nc, /**/ uu));
        break;
    case TYPE_CIRCLE:
        KL(inflow_dev::ini_vel, (k_cnf(n)), (vp->circle, p->circle, nc, /**/ uu));
        break;
    case TYPE_NONE:
        break;
    default:
        ERR("No inflow type is set");
        break;
    };
}

void inflow_ini_velocity(Inflow *i) {
    ini_velocity(i->t, i->d.nc, &i->p, &i->vp, /**/ i->d.uu);
}

void inflow_fin(Inflow *i) {
    Desc *d = &i->d;
    Dfree(d->rnds);
    Dfree(d->uu);
    Dfree(d->cumflux);
    Dfree(d->ndev);
    EFREE(i);
}

static void create_solvent(float kBT, int numdensity, float dt, Inflow *i, int *n, SolventWrap wrap) {
    int2 nc;
    Desc *d;
    int nctot;
    float nxdt;

    d = &i->d;
    nc = d->nc;
    nctot = nc.x * nc.y;
    nxdt = numdensity * dt;

    CC(d::MemcpyAsync(d->ndev, n, sizeof(int), H2D));

    switch(i->t) {
    case TYPE_PLATE:
        KL(inflow_dev::cumulative_flux, (k_cnf(nctot)), (nxdt, i->p.plate, nc, d->uu, /**/ d->cumflux));
        KL(inflow_dev::create_particles, (k_cnf(nctot)),
           (kBT, i->p.plate, nc, d->uu, /*io*/ d->rnds, d->cumflux, /**/ d->ndev, wrap));
        break;
    case TYPE_CIRCLE:
        KL(inflow_dev::cumulative_flux, (k_cnf(nctot)), (nxdt, i->p.circle, nc, d->uu, /**/ d->cumflux));
        KL(inflow_dev::create_particles, (k_cnf(nctot)),
           (kBT, i->p.circle, nc, d->uu, /*io*/ d->rnds, d->cumflux, /**/ d->ndev, wrap));
        break;
    case TYPE_NONE:
        break;
    default:
        ERR("No inflow type is set");
        break;
    };

    CC(d::MemcpyAsync(n, d->ndev, sizeof(int), D2H));
    dSync(); // wait for n
}

void inflow_create_pp(float kBT, int numdensity, float dt, Inflow *i, int *n, Particle *pp) {
    SolventWrap wrap;

    wrap.pp = pp;
    wrap.cc = NULL;
    wrap.multisolvent = false;
    wrap.color = -1;

    create_solvent(kBT, numdensity, dt, i, n, wrap);
}

void inflow_create_pp_cc(float kBT, int numdensity, float dt, int newcolor, Inflow *i, int *n, Particle *pp, int *cc) {
    SolventWrap wrap;

    wrap.pp = pp;
    wrap.cc = cc;
    wrap.multisolvent = true;
    wrap.color = newcolor;
    create_solvent(kBT, numdensity, dt, i, n, wrap);
}
/* end src/control/inflow/imp/main.h */
/* end src/control/inflow/imp.cu */

#line 1 "src/control/outflow/conf.cpp"

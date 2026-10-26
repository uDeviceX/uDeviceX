// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/scheme/force/conf.cpp */
#include <stdlib.h>
#include <stdio.h>
#include <vector_types.h>

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

/* begin src/scheme/force/imp.h */
struct Particle;
struct Force;
struct BForce;
struct float3;
struct Config;
struct Coords;

// tag::mem[]
void bforce_ini(BForce **p);
void bforce_fin(BForce *p);
// end::mem[]

// tag::set[]
void bforce_set_none(/**/ BForce *p);            // <1>
void bforce_set_cste(float3 f, /**/ BForce *p);  // <2>
void bforce_set_dp(float a, /**/ BForce *p);     // <3>
void bforce_set_shear(float a, /**/ BForce *p);  // <4>
void bforce_set_rol(float a, /**/ BForce *p);    // <5>
void bforce_set_rad(float a, /**/ BForce *p);    // <6>
// end::set[]

// tag::cnf[]
void bforce_set_conf(const Config *cfg, /**/ BForce *bf);
// end::cnf[]

// tag::interface[]
void bforce_adjust(float3 f, /**/ BForce *fpar); // <1>
void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff); // <2>
// end::interface[]
/* end src/scheme/force/imp.h */

void bforce_set_conf(const Config *cfg, /**/ BForce *bf) {
    const char *type;
    UC(conf_lookup_string(cfg, "bforce.type", /**/ &type));

    if      (same_str(type, "none")) {
        bforce_set_none(/**/ bf);
    }
    else if (same_str(type, "constant")) {
        float3 f;
        UC(conf_lookup_float3(cfg, "bforce.f", /**/ &f));
        UC(bforce_set_cste(f, /**/ bf));
    }
    else if (same_str(type, "double_poiseuille")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_dp(a, /**/ bf));
    }
    else if (same_str(type, "shear")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_shear(a, /**/ bf));
    }
    else if (same_str(type, "four_roller")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_rol(a, /**/ bf));
    }
    else if (same_str(type, "rad")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_set_rad(a, /**/ bf));
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

/* end src/scheme/force/conf.cpp */

#line 1 "src/scheme/force/imp.cu"


/* begin src/scheme/force/imp.cu */
#include <stdio.h>

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
#ifndef UDX_SCHEME_TYPE_COMMON
#define UDX_SCHEME_TYPE_COMMON
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

/* begin src/d/api.h */
#ifndef UDX_SCHEME_DEVICE_API_COMMON
#define UDX_SCHEME_DEVICE_API_COMMON
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
#ifndef UDX_SCHEME_KL_COMMON
#define UDX_SCHEME_KL_COMMON
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
#ifndef UDX_SCHEME_CC_COMMON
#define UDX_SCHEME_CC_COMMON
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



/* begin src/scheme/force/type.h */
/* "view" structures: to be passed to device funtions */

/* constant force f */
struct BForce_cste_v {
    float3 a;
};

/* double poiseuille */
struct BForce_dp_v {
    float a;
};

/* shear force fx = a * (y - yc) */
struct BForce_shear_v {
    float a;
};

/* 4 rollers mill */
struct BForce_rol_v {
    float a;
};

/* radial force decaying as 1/r */
struct BForce_rad_v {
    float a;
};

enum {
    BODY_FORCE_V_NONE,
    BODY_FORCE_V_CSTE,
    BODY_FORCE_V_DP,
    BODY_FORCE_V_SHEAR,
    BODY_FORCE_V_ROL,
    BODY_FORCE_V_RAD
};

union BForceParam_v {
    BForce_cste_v cste;
    BForce_dp_v dp;
    BForce_shear_v shear;
    BForce_rol_v rol;
    BForce_rad_v rad;
};

struct BForce_v {
    BForceParam_v p;
    int type;
};
/* end src/scheme/force/type.h */
/* begin src/scheme/force/dev/main.h */
/* cste */
static __device__ float3 get_gp(Coords_v, BForce_cste_v par, Particle) {
    return par.a;
}

/* double poiseuille */
static __device__ float3 get_gp(Coords_v c, BForce_dp_v par, Particle p) {
    enum {X, Y, Z};
    float d, f;
    d = yl2yc(c, p.r[Y]);
    f = d > 0 ? par.a : -par.a;
    return make_float3(f, 0, 0);
}

/* shear */
static __device__ float3 get_gp(Coords_v c, BForce_shear_v par, Particle p) {
    enum {X, Y, Z};
    float d, f;
    d = yl2yc(c, p.r[Y]);
    f = d * par.a;
    return make_float3(f, 0, 0);
}

/* 4 roller mills */
static __device__ float3 get_gp(Coords_v c, BForce_rol_v par, Particle p) {
    enum {X, Y, Z};
    float x, y, lx, ly;
    float3 f;
    const float PI = 3.141592653589793;

    lx = xdomain(c);
    ly = ydomain(c);

    x = xl2xg(c, p.r[X]);
    y = yl2yg(c, p.r[Y]);

    x *= 2*PI / lx;
    y *= 2*PI / ly;

    f.x =  2*sin(x)*cos(y) * par.a;
    f.y = -2*cos(x)*sin(y) * par.a;
    f.z = 0;
    return f;
}

/* radial */
static __device__ float3 get_gp(Coords_v c, BForce_rad_v par, Particle p) {
    enum {X, Y, Z};
    float x, y, fact;
    float3 f;

    x = xl2xc(c, p.r[X]);
    y = yl2yc(c, p.r[Y]);

    fact = par.a / (x*x + y*y);

    f.x = fact * x;
    f.y = fact * y;
    f.z = 0;
    return f;
}

/* common */
template <typename P>
__global__ void force(Coords_v c, P par, float mass, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y, Z};
    int i;
    Particle p;
    float3 gp; // pressure gradient
    float *f; // force
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    p = pp[i];
    f = ff[i].f;
    gp = get_gp(c, par, p);

    f[X] += mass * gp.x;
    f[Y] += mass * gp.y;
    f[Z] += mass * gp.z;
}
/* end src/scheme/force/dev/main.h */
/* begin src/scheme/force/imp/type.h */
/* structure containing parameters on host */

// tag::params[]
/* constant force f */
struct BForce_cste {
    float3 a; // acceleration vector
};

/* double poiseuille */
struct BForce_dp {
    float a; // acceleration in x direction
};

/* shear force fx = a * (y - yc) */
struct BForce_shear {
    float a; // acceleration is a * y in x direction
};

/* 4 rollers mill */
struct BForce_rol {
    float a;  // intensity of the acceleration
};

/* radial force decaying as 1/r */
struct BForce_rad {
    float a; // radial acceleartion is a / r
};
// end::params[]

enum {
    BODY_FORCE_NONE,
    BODY_FORCE_CSTE,
    BODY_FORCE_DP,
    BODY_FORCE_SHEAR,
    BODY_FORCE_ROL,
    BODY_FORCE_RAD
};

union BForceParam {
    BForce_cste cste;
    BForce_dp dp;
    BForce_shear shear;
    BForce_rol rol;
    BForce_rad rad;
};

struct BForce {
    BForceParam p;
    int type;
};
/* end src/scheme/force/imp/type.h */
/* begin src/scheme/force/imp/ini.h */
void bforce_ini(BForce **p) { EMALLOC(1, p); }
void bforce_fin(BForce *p)  { EFREE(p);      }

void bforce_set_none(/**/ BForce *p) {
    p->type = BODY_FORCE_NONE;
}

void bforce_set_cste(float3 f, /**/ BForce *p) {
    p->type = BODY_FORCE_CSTE;
    p->p.cste.a = f;
}

void bforce_set_dp(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_DP;
    p->p.dp.a = a;
}

void bforce_set_shear(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_SHEAR;
    p->p.shear.a = a;
}

void bforce_set_rol(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_ROL;
    p->p.rol.a = a;
}

void bforce_set_rad(float a, /**/ BForce *p) {
    p->type = BODY_FORCE_RAD;
    p->p.rad.a = a;
}
/* end src/scheme/force/imp/ini.h */
/* begin src/scheme/force/imp/main.h */
void bforce_adjust(float3 f, /**/ BForce *bforce) {
    int type;
    BForceParam *p;
    type = bforce->type;
    p    = &bforce->p;

    switch (type) {
    case BODY_FORCE_NONE:
        break;
    case BODY_FORCE_CSTE:
        p->cste.a = f;
        break;
    case BODY_FORCE_RAD:
        /* do not control radial and z directions */
        p->rad.a = f.x;
        break;
    case BODY_FORCE_DP:
    case BODY_FORCE_SHEAR:
    case BODY_FORCE_ROL:
    default:
        ERR("not implemented");
        break;
    };
}

static void get_view(BForce_cste bf, BForce_v *v) {
    v->type = BODY_FORCE_V_CSTE;
    v->p.cste.a = bf.a;
}

static void get_view(BForce_dp bf, BForce_v *v) {
    v->type = BODY_FORCE_V_DP;
    v->p.dp.a = bf.a;
}

static void get_view(BForce_shear bf, BForce_v *v) {
    v->type = BODY_FORCE_V_SHEAR;
    v->p.shear.a = bf.a;
}

static void get_view(BForce_rol bf, BForce_v *v) {
    v->type = BODY_FORCE_V_ROL;
    v->p.rol.a = bf.a;
}

static void get_view(BForce_rad bf, BForce_v *v) {
    v->type = BODY_FORCE_V_RAD;
    v->p.rad.a = bf.a;
}

static void bforce_get_view(const BForce *bforce, /**/ BForce_v *view) {
    int type;
    BForceParam p;
    type = bforce->type;
    p    = bforce->p;

    switch (type) {
    case BODY_FORCE_NONE:
        view->type = BODY_FORCE_V_NONE;
        break;
    case BODY_FORCE_CSTE:
        get_view(p.cste, /**/ view);
        break;
    case BODY_FORCE_DP:
        get_view(p.dp, /**/ view);
        break;
    case BODY_FORCE_SHEAR:
        get_view(p.shear, /**/ view);
        break;
    case BODY_FORCE_ROL:
        get_view(p.rol, /**/ view);
        break;
    case BODY_FORCE_RAD:
        get_view(p.rad, /**/ view);
        break;
    default:
        ERR("wrong type");
        break;
    };
}

void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff) {
    int type;
    BForce_v view;
    BForceParam_v p;
    Coords_v coordsv;

    coords_get_view(c, &coordsv);

    view.type = BODY_FORCE_V_NONE;
    bforce_get_view(bf, /**/ &view);

    type = view.type;
    p    = view.p;

    switch (type) {
    case BODY_FORCE_V_NONE:
        break;
    case BODY_FORCE_V_CSTE:
        KL(force, (k_cnf(n)), (coordsv, p.cste, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_DP:
        KL(force, (k_cnf(n)), (coordsv, p.dp, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_SHEAR:
        KL(force, (k_cnf(n)), (coordsv, p.shear, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_ROL:
        KL(force, (k_cnf(n)), (coordsv, p.rol, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_RAD:
        KL(force, (k_cnf(n)), (coordsv, p.rad, mass, n, pp, /**/ ff));
        break;
    default:
        ERR("wrong type <%d>", type);
        break;
    };
}
/* end src/scheme/force/imp/main.h */
/* end src/scheme/force/imp.cu */

#line 1 "src/scheme/move/imp.cu"

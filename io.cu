
// Editable compilation unit for scheme/system and I/O clients.


// Grouped editable compilation unit for scheme and simulation-system code.


// Grouped editable compilation unit for force, time, move, and restraint schemes.


// Grouped editable compilation unit for force and time schemes.


#include "config.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <vector_types.h>

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

struct Particle;
struct Force;
struct BForce;
struct float3;
struct Config;
struct Coords;

void bforce_ini(BForce **p);
void bforce_fin(BForce *p);

void bforce_set_none(/**/ BForce *p);            // <1>
void bforce_set_cste(float3 f, /**/ BForce *p);  // <2>
void bforce_set_dp(float a, /**/ BForce *p);     // <3>
void bforce_set_shear(float a, /**/ BForce *p);  // <4>
void bforce_set_rol(float a, /**/ BForce *p);    // <5>
void bforce_set_rad(float a, /**/ BForce *p);    // <6>

void bforce_set_conf(const Config *cfg, /**/ BForce *bf);

void bforce_adjust(float3 f, /**/ BForce *fpar); // <1>
void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff); // <2>

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

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
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

struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
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

#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
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
#ifndef UDX_SCHEME_KL_COMMON
#define UDX_SCHEME_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_SCHEME_CC_COMMON
#define UDX_SCHEME_CC_COMMON
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
#endif



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
/* structure containing parameters on host */

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




// Grouped editable compilation unit for time-line and time-step schemes.



#include <mpi.h>
#include <stdio.h>
#include <math.h>

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

struct Coords;
struct Particle;
struct Rigid;

#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
#endif

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

struct TimeLine;

void time_line_ini(float start, /**/ TimeLine**);
void time_line_fin(TimeLine*);

void time_line_advance(float dt, TimeLine*);           // <1>
int  time_line_cross(const TimeLine*, float interval); // <2>

void time_line_strt_dump(MPI_Comm, const char *base, int id, const TimeLine*); // <1>
void time_line_strt_read(const char *base, int id, TimeLine*); // <2>

float time_line_get_current  (const TimeLine*); // <1>
long  time_line_get_iteration(const TimeLine*); // <2>

struct TimeLine {
    float t, t0;
    float dt, dt0;
    long iteration;
};
void time_line_ini(float t, /**/ TimeLine** pq) {
    TimeLine *q;
    EMALLOC(1, &q);
    q->t = q->t0 = t;
    q->iteration = 0;
    *pq = q;
}
void time_line_fin(TimeLine *q) { EFREE(q); }

void time_line_advance(float dt, TimeLine *q) {
    if (dt < 0)
        ERR("dt = %g < 0", dt);
    if (q->iteration == 0) q->dt0 = dt;
    else                   q->t0 = q->t;

    q->t  += dt;
    q->dt0 = q->dt;
    q->dt  = dt;

    q->iteration++;
}

float time_line_get_current(const TimeLine *q) { return q->t; }
long  time_line_get_iteration(const TimeLine *q) { return q->iteration; }
int time_line_cross(const TimeLine *q, float i) {
    float t, t0, f;
    if (q->iteration == 0) return 0;
    else if (i < 0)   ERR("i < 0");
    else if (i == 0)  return 1;

    t = q->t; t0 = q->t0;
    f = floor(t / i) * i;
    return t0 <= f && f < t;
}
#define CODE "timeline"

static void write_state(const TimeLine *t, FILE *f) {
    fprintf(f, "%g %g\n", t->t,  t->t0);
    fprintf(f, "%g %g\n", t->dt, t->dt0);
    fprintf(f, "%ld\n", t->iteration);
}

static void stream_write_state(const void *data, FILE *f) {
    const TimeLine *t = (const TimeLine*) data;
    write_state(t, f);
}

void time_line_strt_dump(MPI_Comm comm, const char *base, int id, const TimeLine *t) {
    StreamWriter sw = stream_write_state;
    restart_write_stream_one_node(comm, base, CODE, id, (void*) t, sw);
}

static void read_state(FILE *f, TimeLine *t) {
    fscanf(f, "%g %g\n", &t->t,  &t->t0);
    fscanf(f, "%g %g\n", &t->dt, &t->dt0);
    fscanf(f, "%ld\n", &t->iteration);
}

static void stream_read_state(FILE *f, void *data) {
    TimeLine *t = (TimeLine*) data;
    UC(read_state(f, t));
}

void time_line_strt_read(const char *base, int id, TimeLine *t) {
    StreamReader sr = stream_read_state;
    restart_read_stream_one_node(base, CODE, id, sr, (void*) t);
}

#undef CODE





#include <mpi.h>
#include <math.h>

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
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

struct Force;

/* sqrt(max(fx^2 + fy^2 + fz^2)) */
float force_stat_max(int n, const Force*);
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

struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

void time_step_accel_ini(/**/ TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);

void time_step_accel_push(TimeStepAccel*, float m, int n, Force*); // <1>
void time_step_accel_reset(TimeStepAccel*); // <2>


void time_step_ini(const Config*, /**/ TimeStep**); // <1>
void time_step_fin(TimeStep*);

float time_step_dt(TimeStep*, MPI_Comm, TimeStepAccel*); // <1>
float time_step_dt0(TimeStep*);                          // <2>
void time_step_log(TimeStep*);                           // <3>

enum { MAX_ACCEL_NUM = 9 };
struct TimeStepAccel {
    float  mm[MAX_ACCEL_NUM];
    int    nn[MAX_ACCEL_NUM];
    Force *fff[MAX_ACCEL_NUM];
    int k; /* number of forces */
};

struct TimeStep {
    int type;
    bool screenlog;
    float dt;
    float dx;

    /* to log */
    float accel[MAX_ACCEL_NUM];
    int k;
};

enum {CONST, DISP};
static float const_dt(TimeStep*, MPI_Comm, TimeStepAccel*);
static float disp_dt(TimeStep*, MPI_Comm, TimeStepAccel*);
static void const_log(TimeStep*);
static void  disp_log(TimeStep*);
static void  (*tlog[])(TimeStep*)                          = { const_log,  disp_log };
static float (*dt[])(TimeStep*, MPI_Comm, TimeStepAccel*) = { const_dt,  disp_dt };
#ifndef UDX_SCHEME_IO_REDUCE_COMMON
#define UDX_SCHEME_IO_REDUCE_COMMON
static void reduce(MPI_Comm comm, const void *sendbuf0, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    int root = 0;
    const void *sendbuf = (m::is_master(comm) ? MPI_IN_PLACE : sendbuf0);
    MC(m::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm));
}
#endif /* UDX_SCHEME_IO_REDUCE_COMMON */

static void max_float(MPI_Comm comm, /*io*/ float *v) {
    int count = 1;
    UC(reduce(comm, v, v, count, MPI_FLOAT, MPI_MAX));
}

static float accel_max(MPI_Comm comm, TimeStepAccel *q, /**/ float *accel) {
    int n, i;
    float force, mass, max;
    Force *ff;
    max = 0;
    for (i = 0; i < q->k; i++) {
        mass = q->mm[i]; n = q->nn[i]; ff = q->fff[i];
        force = force_stat_max(n, ff);
        accel[i] = force/mass;
        if (accel[i] > max) max = accel[i];
    }
    UC(max_float(comm, &max));
    return max;
}
void time_step_accel_ini(/**/ TimeStepAccel **pq) {
    TimeStepAccel *q;
    EMALLOC(1, &q);
    q->k = 0;
    *pq = q;
}
void time_step_accel_fin(TimeStepAccel *q) { EFREE(q); }
void time_step_accel_push(TimeStepAccel *q, float m, int n, Force *ff) {
    int k;
    k = q->k;
    if (k + 1 == MAX_ACCEL_NUM)
        ERR("k + 1 = %d < %d = MAX_ACCEL_NUM", k + 1, MAX_ACCEL_NUM);
    if (m <= 0) ERR("m <= 0");
    q->mm[k] = m; q->nn[k] = n; q->fff[k] = ff;
    q->k = k + 1;
}
void time_step_accel_reset(TimeStepAccel *q) { q->k = 0; }

void time_step_ini(const Config *c, /**/ TimeStep **pq) {
    const char *type;
    int l;
    TimeStep *q;
    EMALLOC(1, &q);

    UC(conf_lookup_string(c, "time.type", &type));
    if      (same_str(type, "const")) {
        q->type = CONST;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
    } else if (same_str(type, "disp")) {
        q->type = DISP;
        q->k    = 0;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
        UC(conf_lookup_float(c, "time.dx", &q->dx));
    } else
        ERR("unrecognised time.type: '%s'", type);

    UC(conf_lookup_bool(c, "time.screenlog", &l));
    q->screenlog = l;

    *pq = q;
}
void time_step_fin(TimeStep *q) {EFREE(q); }

static float const_dt(TimeStep *q, MPI_Comm, TimeStepAccel*) { return q->dt; }
static float disp_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    float dt_new, dx, dt_max, accel;
    dt_max = q->dt;
    dx     = q->dx;
    accel = accel_max(comm, a, /**/ q->accel);
    q->k  = a->k;
    if (accel == 0) return dt_max;
    dt_new = sqrtf(2.*dx/accel);
    return (dt_new > dt_max ? dt_max : dt_new);
}
float time_step_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    return dt[q->type](q, comm, a);
}

static void const_log(TimeStep *q) { msg_print(" time_step: const: dt=%.4g\n", q->dt); }
static void  disp_log(TimeStep *q) {
    int i;
    float accel;
    float dx;
    dx = q->dx;
    msg_print(" time_step: disp: dt=%.4e, dx=%.4e", q->dt, q->dx);
    for (i = 0; i < q->k; i++) {
        accel = q->accel[i];
        msg_print(" accel:%.4e, dt_acc:%.4e, dt_used:%.4e",
                  accel, sqrtf(2.*dx/accel), (sqrtf(2.*dx/accel) > q->dt ? q->dt : sqrtf(2.*dx/accel)));
    }
    msg_print("");
}
void time_step_log(TimeStep *q) { if (q->screenlog==true) tlog[q->type](q); }
float time_step_dt0(TimeStep *q) { return q->dt; };






// Grouped editable compilation unit for move and restraint schemes.



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


#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
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

#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
#ifndef UDX_SCHEME_CC_COMMON
#define UDX_SCHEME_CC_COMMON
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
#endif

#ifndef UDX_SCHEME_KL_COMMON
#define UDX_SCHEME_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
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

struct Force;
struct Particle;

void scheme_move_apply(float dt, float mass, int n, const Force*, Particle*); // <1>
void scheme_move_clear_vel(int n, Particle*); // <2>

namespace scheme_move_dev {
#ifdef FORWARD_EULER
static __device__ void update0(float dt, float m, const float *f, /**/ float *r, float *v) {
    enum {X, Y, Z};
    r[X] += v[X]*dt;
    r[Y] += v[Y]*dt;
    r[Z] += v[Z]*dt;

    v[X] += f[X]/m*dt;
    v[Y] += f[Y]/m*dt;
    v[Z] += f[Z]/m*dt;
}
#else
static __device__ void update0(float dt, float m, const float *f, /**/ float *r, float *v) {
    enum {X, Y, Z};
    v[X] += f[X]/m*dt;
    v[Y] += f[Y]/m*dt;
    v[Z] += f[Z]/m*dt;

    r[X] += v[X]*dt;
    r[Y] += v[Y]*dt;
    r[Z] += v[Z]*dt;
}
#endif
__global__ void update(float dt, float mass, Particle* pp, const Force *ff, int n) {
    float *r, *v;
    const float *f;
    int pid;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    v = pp[pid].v;
    f = ff[pid].f;
    update0(dt, mass, f, /**/ r, v);
}

__global__ void clear_vel(int n, /**/ Particle *pp)  {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    for(int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
}
} /* namespace */

void scheme_move_apply(float dt, float mass, int n, const Force *ff, Particle *pp) {
    KL(scheme_move_dev::update, (k_cnf(n)), (dt, mass, pp, ff, n));
}

void scheme_move_clear_vel(int n, /**/ Particle *pp) {
    KL(scheme_move_dev::clear_vel, (k_cnf(n)), (n, /**/ pp));
}





#include <stdio.h>
#include <mpi.h>

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

struct Particle;
struct Config;
struct Restrain;

struct SchemeQQ {
    Particle *o, *r;
    int on, rn;
};

void scheme_restrain_ini(Restrain**);
void scheme_restrain_fin(Restrain*);

void scheme_restrain_set_red(Restrain*); // <1>
void scheme_restrain_set_rbc(Restrain*); // <2>
void scheme_restrain_set_none(Restrain*); // <3>
void scheme_restrain_set_freq(int freq, Restrain*); // <4>

void scheme_restrain_set_conf(const Config*, Restrain*);

void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it, /**/ Restrain *r, SchemeQQ qq);

void scheme_restrain_set_conf(const Config *cfg, Restrain *r) {
    const char *kind;
    int freq;

    UC(conf_lookup_string(cfg, "restrain.kind", &kind));
    UC(conf_lookup_int(cfg, "restrain.freq", &freq));

    scheme_restrain_set_freq(0, r);

    if      (same_str(kind, "none"))
        scheme_restrain_set_none(r);
    else if (same_str(kind, "rbc"))
        scheme_restrain_set_rbc(r);
    else if (same_str(kind, "red"))
        scheme_restrain_set_red(r);
    else
        ERR("Unrecognise kind <%s>", kind);
}



#include <mpi.h>
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


#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
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
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
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
#ifndef UDX_SCHEME_CC_COMMON
#define UDX_SCHEME_CC_COMMON
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
#endif
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
#ifndef UDX_SCHEME_KL_COMMON
#define UDX_SCHEME_KL_COMMON
#include "launch.cuh"
#endif
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


#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
#endif



namespace restrain_dev {
struct MapColor {
    const int *cc;
    int color;
};

struct MapGrey {};
static __device__ int valid_p(const MapGrey, int) { return 1; }
static __device__ int valid_p(const MapColor m, int i) {
    return m.cc[i] == m.color;
}

template <typename Map>
__global__ void sum(Map m, int n, const Particle *pp, /**/ int *ntot, float3 *vtot) {
    enum {X, Y, Z};
    int i, good, ngood;
    float3 v;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    good = (i < n) && valid_p(m, i);

    if (good) {
        const Particle p = pp[i];
        v.x = p.v[X];
        v.y = p.v[Y];
        v.z = p.v[Z];
    } else {
        v.x = v.y = v.z = 0;
    }

    v     = warpReduceSum(v);
    ngood = warpReduceSum(good);

    if ((threadIdx.x % warpSize == 0) && ngood > 0) {
        atomicAdd(&vtot->x, v.x);
        atomicAdd(&vtot->y, v.y);
        atomicAdd(&vtot->z, v.z);
        atomicAdd(ntot, ngood);
    }
}

template <typename Map>
__global__ void shift(Map m, float3 v, int n, /**/ Particle *pp) {
    enum {X, Y, Z};
    int i, good;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    good = (i < n) && valid_p(m, i);

    if (good) {
        Particle p = pp[i];
        p.v[X] -= v.x;
        p.v[Y] -= v.y;
        p.v[Z] -= v.z;
        pp[i] = p;
    }
}
}

enum {
    RSTR_NONE,
    RSTR_COL,
    RSTR_RBC
};

struct Restrain {
    int kind;
    int freq;

    /* pinned */
    int *n;
    float3 *v;
};
void scheme_restrain_ini(Restrain **rstr) {
    Restrain *r;
    EMALLOC(1, rstr);
    r = *rstr;

    CC(d::alloc_pinned((void**) &r->n, sizeof(int)));
    CC(d::alloc_pinned((void**) &r->v, sizeof(float3)));
}

void scheme_restrain_fin(Restrain *r) {
    CC(d::FreeHost(r->n));
    CC(d::FreeHost(r->v));
    EFREE(r);
}

void scheme_restrain_set_red(Restrain *r) {
    r->kind = RSTR_COL;
}

void scheme_restrain_set_rbc(Restrain *r) {
    r->kind = RSTR_RBC;
}

void scheme_restrain_set_none(Restrain *r) {
    r->kind = RSTR_NONE;
}

void scheme_restrain_set_freq(int freq, Restrain *r) {
    r->freq = freq;
}

static const char* kind2tag(int i) {
    static const char *tags[] = {"NONE", "RED", "RBC"};
    return tags[i];
}

static void print_stats(const char *tag, int n, float3 v) {
    msg_print("restrain %s: n = %d [% .3e % .3e % .3e]", tag, n, v.x, v.y, v.z);
}

static void report(const Restrain *r, long it) {
    int n, freq, cond, kind;
    float3 v;
    kind = r->kind;
    freq = r->freq;
    cond = freq > 0 && it % freq == 0 && kind != RSTR_NONE;
    n = *(r->n);
    v = *(r->v);
    if (cond)
        print_stats(kind2tag(kind), n, v);
}

static void reini(Restrain *r) {
    *(r->n) = 0;
    *(r->v) = make_float3(0, 0, 0);
}

static int all_reduce(MPI_Comm comm, int n) {
    int ntot = 0;
    MC(m::Allreduce(&n, &ntot, 1, MPI_INT, MPI_SUM, comm));
    return ntot;
}

static float3 all_reduce(MPI_Comm comm, float3 v) {
    float3 vtot = make_float3(0, 0, 0);
    MC(m::Allreduce(&v.x, &vtot.x, 3, MPI_FLOAT, MPI_SUM, comm));
    return vtot;
}

template <typename Map>
static void apply(MPI_Comm comm, Map m, int np, /**/ Particle *pp, Restrain *r) {
    int n;
    float3 v;
    float scale;

    reini( /**/ r);
    KL(restrain_dev::sum, (k_cnf(np)), (m, np, pp, /**/ r->n, r->v));
    dSync(); /* wait for n and v to be computed */

    n = *(r->n);
    v = *(r->v);

    n = all_reduce(comm, n);
    v = all_reduce(comm, v);

    scale = n ? 1.0 / n : 1;
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;

    KL(restrain_dev::shift, (k_cnf(np)), (m, v, np, /**/ pp));

    *(r->v) = v;
    *(r->n) = n;
}

void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it, /**/ Restrain *r, SchemeQQ qq) {
    int kind;
    kind = r->kind;
    switch (kind) {
    case RSTR_NONE:
        break;
    case RSTR_COL:
        restrain_dev::MapColor mc;
        mc.cc = cc;
        mc.color = RED_COLOR;
        apply(comm, mc, qq.on, /**/ qq.o, r);
        report(r, it);
        break;
    case RSTR_RBC:
        restrain_dev::MapGrey mg;
        apply(comm, mg, qq.rn, /**/ qq.r, r);
        report(r, it);
        break;
    default:
        ERR("Unknown kind <%d>", kind);
    };
}






// Grouped editable compilation unit for simulation options, objects, and walls.


// Grouped editable compilation unit for simulation options and objects.


// Grouped editable compilation unit for simulation objects and interactions.



#include <mpi.h>
#include <stdlib.h>
#include <string.h>
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

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
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

#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
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
void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
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
struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;
struct Coords;

enum {
    DBG_POS,      // <1>
    DBG_POS_SOFT, // <2>
    DBG_VEL,      // <3>
    DBG_FORCES,   // <4>
    DBG_COLORS,   // <5>
    DBG_CLIST,    // <6>
    DBG_NKIND_
};

void dbg_ini(Dbg**);
void dbg_fin(Dbg*);

void dbg_enable(int kind, Dbg *dbg);
void dbg_disable(int kind, Dbg *dbg);
void dbg_set_verbose(bool, Dbg *dbg);
void dbg_set_dump(bool, Dbg *dbg);

void dbg_set_conf(const Config*, Dbg*);

void dbg_check_pos(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_pos_soft(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_vel(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_forces(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp, const Force *ff);
void dbg_check_colors(const Coords *c, const Dbg *dbg, int n, const int *cc);
void dbg_check_clist(const Coords *c, const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp);

void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
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
struct Particle;
struct Positioncp;
struct float4;

void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr);
void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr);
void convert_pp2f4_pos(int n, const float *pp, /**/ float4 *zpp);

struct Coords;
struct Config;

void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c); // <1>
void coords_fin(Coords *c); // <2>

void coords_ini_conf(MPI_Comm cart, const Config *cfg, Coords **c);

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
#ifndef UDX_SIM_OBJECTS_PARRAY_COMMON
#define UDX_SIM_OBJECTS_PARRAY_COMMON
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
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
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
#endif
#ifndef UDX_SIM_OBJECTS_PFARRAYS_COMMON
#define UDX_SIM_OBJECTS_PFARRAYS_COMMON
struct PFarray {
    PaArray p;
    FoArray f;
    long n;
};

struct PFarrays;

struct PaArray;
struct FoArray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarrays_clear(PFarrays*);
void pfarrays_push(PFarrays*, long n, PaArray, FoArray);
void pfarrays_push(PFarrays*, PFarray);
int  pfarrays_size(const PFarrays*);
void pfarrays_get(int i, const PFarrays*, long *n, PaArray*, FoArray*);
void pfarrays_get(int i, const PFarrays*, PFarray*);
#endif
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
struct Triangles {
    int4 *tt;
    int nt;
};
struct Triangles;
struct MeshRead;

void triangles_ini(MeshRead*, /**/ Triangles**);
void triangles_fin(Triangles*);
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

struct Particle;
struct float3;
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);

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
struct Coords;
struct float3;
void io_com_dump(MPI_Comm, const Coords*, const char *name, long id, int n, const int *ii, const float3 *rr);
struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
#ifndef UDX_SIM_MESH_IO_COMMON
#define UDX_SIM_MESH_IO_COMMON
struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

#ifndef UDX_SIM_MESH_WRITE_COMMON
#define UDX_SIM_MESH_WRITE_COMMON
enum {MESH_WRITE_EDGE, MESH_WRITE_CENTER};
#endif

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory, /**/ MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
#endif
#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
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
#endif
struct Coords;
struct Rigid;

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Rigid *ss);

struct IoBop;

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **t);
void io_bop_fin(IoBop *t);

void io_bop_parts       (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t);
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t);
void io_bop_stresses    (MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t);
void io_bop_ids         (MPI_Comm cart, long n, const int   *ii, const char *name, int id, IoBop *t);
void io_bop_colors      (MPI_Comm cart, long n, const int   *cc, const char *name, int id, IoBop *t);

struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
    float totArea, totVolume;
};

void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);
void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *);

double rbc_params_get_tot_volume(const RbcParams *);

void rbc_params_set_conf(const Config *c, const char *name, RbcParams *);

RbcParams_v rbc_params_get_view(const RbcParams *);
struct AreaVolume;
struct Particle;

struct RbcQuants {
    int n, nc;               /* number of particles, cells                */
    int nt, nv;              /* number of triangles and vertices per mesh */
    int md;                  /* maximum valence of the mesh               */
    Particle *pp, *pp_hst;   /* vertices particles on host and device     */
    bool ids;                /* global ids active ?                       */
    int *ii;                 /* global ids on host                        */
    AreaVolume *area_volume; /* to compute area and volume */
};

struct Coords;
struct MeshRead;

void rbc_ini(long maxnc, bool ids, const MeshRead*, RbcQuants*);
void rbc_fin(RbcQuants*);

void rbc_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic, /**/ RbcQuants*); // <1>
void rbc_gen_freeze(MPI_Comm, /**/ RbcQuants*);                                               // <2>
void rbc_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RbcQuants*);  // <3>
void rbc_strt_dump(MPI_Comm, const char *base, const char *name, int id, const RbcQuants *q); // <4>

struct RbcParams;
struct RbcForce;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;

void rbc_force_ini(const MeshRead*, RbcForce**);
void rbc_force_fin(RbcForce*);

void rbc_force_set_stressful(int nt, float totArea, /**/ RbcForce*); // <1>
void rbc_force_set_stressfree(const char *fname, /**/ RbcForce*);    // <2>

void rbc_force_set_rnd0(RbcForce*);           // <3>
void rbc_force_set_rnd1(int seed, RbcForce*); // <4>
void rbc_force_set_bending(const MeshRead*, const char*, RbcForce*);

void rbc_force_set_conf(const MeshRead*, const Config *cfg, const char *name, RbcForce*);

void rbc_force_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*, /**/ Force*); // <1>
void rbc_force_stat(/**/ float *pArea, float *pVolume); // <2>
struct AreaVolume;
struct Particle;
struct int4;

void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**); // <1>
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*, /**/ float **av); // <2>
void area_volume_host(AreaVolume*, /**/ float **av); // <3>
struct RbcStretch;
struct Force;

void rbc_stretch_ini(const char *path, int nv, /**/ RbcStretch**); /* `nv` is for error check */
void rbc_stretch_fin(RbcStretch*);
void rbc_stretch_apply(int nm, const RbcStretch*, /**/ Force*);
struct RbcCom;
struct float3;
struct Particle;

void rbc_com_ini(int nv, int nm_max, /**/ RbcCom**);
void rbc_com_fin(/**/ RbcCom*);
void rbc_com_apply(RbcCom*, int nm, const Particle*, /**/ float3 **rr, float3 **vv);

struct RigQuants {
    int n, ns, nps;              /* number of particles (total), solid, particle per solid        */
    Particle *pp_hst, *pp;       /* particles on hst and device                                   */
    Rigid    *ss_hst, *ss;       /* rigid strutures                                               */
    float   *rr0_hst, *rr0;      /* frozen particle templates                                     */

    /* mesh related quantities */
    int nt, nv;                   /* number of [t]riangles and [v]ertices                          */
    int4 *dtt;                    /* triangle indices on device                                    */
    float *dvv;                   /* vertices on device (template)                                 */
    Particle *i_pp_hst, *i_pp;    /* particles representing all meshes of all solids of that node  */

    Rigid *ss_dmp;

    long maxp; /* maximum particle number */
};

struct Coords;
struct RigPinInfo;
struct MeshRead;

void rig_ini(long maxs, long maxp, const MeshRead*, RigQuants*);
void rig_fin(RigQuants *q);

void rig_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic, /**/ RigQuants *); // <1>
void rig_gen_freeze(const Coords*, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo*,
                    MPI_Comm, const MeshRead*, Particle *opp, int *on, RigQuants *q);          // <2>
void rig_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RigQuants *q); // <3>

void rig_strt_dump_templ(MPI_Comm, const char *base, const char *name, const RigQuants *q);         // <1>
void rig_strt_dump(MPI_Comm, const char *base, const char *name, const int id, const RigQuants *q); // <2>
struct RigPinInfo;
struct int3;
struct Config;

enum {NOT_PERIODIC = -1};

void rig_pininfo_ini(RigPinInfo **); // <1>
void rig_pininfo_fin(RigPinInfo *); // <2>
void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *); // <3>
void rig_pininfo_set_pdir(int pdir, RigPinInfo *); // <4>
void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *); // <5>

int rig_pininfo_get_pdir(const RigPinInfo *); // <6>

struct Particle;
struct Rigid;
struct Force;

void rig_reinit_ft(const int ns, /**/ Rigid *ss); // <1>
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns, /**/ Particle *pp, Rigid *ss); // <2>
void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0, /**/ Particle *pp); // <3>
void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv, /**/ Particle *pp); // <4>


#ifndef UDX_SIM_WALLS_COMMON
#define UDX_SIM_WALLS_COMMON
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
#endif

struct Force;
struct Particle;

void scheme_move_apply(float dt, float mass, int n, const Force*, Particle*); // <1>
void scheme_move_clear_vel(int n, Particle*); // <2>
struct Particle;
struct Force;
struct BForce;
struct float3;
struct Config;
struct Coords;

void bforce_ini(BForce **p);
void bforce_fin(BForce *p);

void bforce_set_none(/**/ BForce *p);            // <1>
void bforce_set_cste(float3 f, /**/ BForce *p);  // <2>
void bforce_set_dp(float a, /**/ BForce *p);     // <3>
void bforce_set_shear(float a, /**/ BForce *p);  // <4>
void bforce_set_rol(float a, /**/ BForce *p);    // <5>
void bforce_set_rad(float a, /**/ BForce *p);    // <6>

void bforce_set_conf(const Config *cfg, /**/ BForce *bf);

void bforce_adjust(float3 f, /**/ BForce *fpar); // <1>
void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff); // <2>
struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

void time_step_accel_ini(/**/ TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);

void time_step_accel_push(TimeStepAccel*, float m, int n, Force*); // <1>
void time_step_accel_reset(TimeStepAccel*); // <2>


void time_step_ini(const Config*, /**/ TimeStep**); // <1>
void time_step_fin(TimeStep*);

float time_step_dt(TimeStep*, MPI_Comm, TimeStepAccel*); // <1>
float time_step_dt0(TimeStep*);                          // <2>
void time_step_log(TimeStep*);                           // <3>

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

struct DRbcPack;
struct DRbcComm;
struct DRbcUnpack;

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
struct DRigPack;
struct DRigComm;
struct DRigUnpack;

void drig_pack_ini(int3 L, int maxns, int nv, DRigPack **p);
void drig_comm_ini(MPI_Comm comm, /**/ DRigComm **c);
void drig_unpack_ini(int3 L, int maxns, int nv, DRigUnpack **u);

void drig_pack_fin(DRigPack *p);
void drig_comm_fin(DRigComm *c);
void drig_unpack_fin(DRigUnpack *u);

void drig_build_map(int ns, const Rigid *ss, /**/ DRigPack *p);
void drig_pack(int ns, int nv, const Rigid *ss, const Particle *ipp, /**/ DRigPack *p);
void drig_download(DRigPack *p);

void drig_post_recv(DRigComm *c, DRigUnpack *u);
void drig_post_send(DRigPack *p, DRigComm *c);
void drig_wait_recv(DRigComm *c, DRigUnpack *u);
void drig_wait_send(DRigComm *c);

void drig_unpack_bulk(const DRigPack *p, /**/ RigQuants *q);
void drig_unpack_halo(const DRigUnpack *u, /**/ RigQuants *q);
struct EMeshPack;
struct EMeshComm;
struct EMeshUnpack;
struct EMeshPackM;
struct EMeshCommM;
struct EMeshUnpackM;

/* mesh exchanger */

void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **p);
void emesh_comm_ini(MPI_Comm comm, /**/ EMeshComm **c);
void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **u);

void emesh_pack_fin(EMeshPack *p);
void emesh_comm_fin(EMeshComm *c);
void emesh_unpack_fin(EMeshUnpack *u);

void emesh_build_map(int nm, int nv, const Particle *pp, /**/ EMeshPack *p);

void emesh_pack_pp(int nv, const Particle *pp, /**/ EMeshPack *p);
void emesh_pack_rrcp(int nv, const Positioncp *rr, /**/ EMeshPack *p);
void emesh_download(EMeshPack *p);

void emesh_post_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_post_send(EMeshPack *p, EMeshComm *c);
void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_wait_send(EMeshComm *c);

void emesh_unpack_pp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Particle *pp);
void emesh_unpack_rrcp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Positioncp *rr);

void emesh_get_num_frag_mesh(const EMeshUnpack *u, /**/ int cc[NFRAGS]);

/* optional: (back) momentum exchanger */

void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshPackM **p);
void emesh_commm_ini(MPI_Comm comm, /**/ EMeshCommM **c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshUnpackM **u);

void emesh_packm_fin(EMeshPackM *p);
void emesh_commm_fin(EMeshCommM *c);
void emesh_unpackm_fin(EMeshUnpackM *u);

void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm, /**/ EMeshPackM *p);
void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p);

void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_post_send(EMeshPackM *p, EMeshCommM *c);
void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_wait_send(EMeshCommM *c);

void emesh_upload(EMeshUnpackM *u);
void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u, /**/ Momentum *mm);
//#include "meshbb/imp.h"
struct MeshBB;

struct Particle;
struct Positioncp;
struct Force;
struct Momentum;
struct Rigid;
struct int4;

struct MeshInfo {
    int nt, nv;
    const int4 *tt;
};

void mesh_bounce_ini(int maxpp, /**/ MeshBB **d);
void mesh_bounce_fin(/**/ MeshBB *d);

void mesh_bounce_reini(int n, /**/ MeshBB *d);  // <1>
void mesh_bounce_find_collisions(float dt, int nm, MeshInfo meshinfo, const Positioncp *i_rr, int3 L,
                                 const int *starts, const int *counts, const Particle *pp, const Particle *pp0, /**/ MeshBB *d);  // <2>
void mesh_bounce_select_collisions(int n, /**/ MeshBB *d);  // <3>
void mesh_bounce_bounce(float dt, float mass, int n, const MeshBB *d, MeshInfo meshinfo,
                        const Positioncp *i_rr, const Particle *pp0, /**/ Particle *pp, Momentum *mm);  // <4>

void mesh_bounce_collect_rig_momentum(int ns, MeshInfo meshinfo, const Particle *pp, const Momentum *mm, /**/ Rigid *ss); // <1>
void mesh_bounce_collect_rbc_momentum(float mass, int nc, MeshInfo mi, const Momentum *mm, /**/ Particle *pp);            // <2>
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

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON
/* solvent options */
struct OptFlu {
    bool colors, ids, ss, push;
};

/* common parameters for membranes and rigid objects */
struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};

/* membrane options */
struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};

/* rigid options */
struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct OptDump {
    bool       parts,      mesh,      field,      strt, forces;
    float freq_parts, freq_mesh, freq_field, freq_strt, freq_diag;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
};

struct Opt {
    OptFlu flu;
    OptMbr mbr[MAX_MBR_TYPES];
    OptRig rig[MAX_RIG_TYPES];
    int nmbr, nrig;
    OptWall wall;
    OptParams params;
    OptDump dump;
    bool inflow, outflow, denoutflow, vcon;
    int recolor_freq;
    int sampler_npdump;
    int3 sampler_grid_ref;
    bool restart;
    bool tracers;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
long opt_estimate_maxp(const Opt*);
#endif

struct Objects;

struct Sdf;
struct Config;
struct Coords;
struct Opt;
struct PFarray;
struct PFarrays;
struct BForce;
struct TimeStepAccel;
struct Dbg;
struct IoBop;
struct PairParams;
struct WallRepulsePrm;

void objects_ini(const Config*, const Opt*, MPI_Comm, const Coords*, int maxp, Objects**);
void objects_fin(Objects*);

void objects_save_mesh(Objects*);
void objects_clear_vel(Objects*);         // <1>
void objects_advance(float dt, Objects*); // <2>
void objects_distribute(Objects*);        // <3>
void objects_update_dpd_prms(float dt, float kBT, Objects*); // <4>

void objects_clear_forces(Objects*);                 // <1>
void objects_body_forces(const BForce*, Objects *o); // <2>

void objects_mesh_dump(Objects*);                    // <1>
void objects_diag_dump(float t, Objects*);           // <2>
void objects_part_dump(long id, Objects*, IoBop*);   // <3>

void objects_strt_templ(const char *base, Objects*);         // <4>
void objects_strt_dump(const char *base, long id, Objects*); // <5>

bool objects_have_bounce(const Objects*);

void objects_get_particles_all(Objects*, PFarrays*);    // <1>
void objects_get_particles_mbr(Objects*, PFarrays*);    // <2>
void objects_get_accel(const Objects*, TimeStepAccel*); // <3>

void objects_get_params_fsi(const Objects*, const PairParams*[]);           // <4>
void objects_get_params_adhesion(const Objects*, const PairParams*[]);      // <5>
void objects_get_params_repulsion(const Objects*, const WallRepulsePrm*[]); // <6>
void objects_get_params_cnt(const Objects*, const PairParams***);           // <7>

void objects_gen_mesh(Objects*);                           // <1>
void objects_remove_from_wall(const Sdf *sdf, Objects *o); // <2>
void objects_gen_freeze(PFarray*, Objects*);               // <3>

void objects_restart(Objects*);

void objects_activate_mbr_bounce(Objects*);
void objects_bounce(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, Objects *obj); // <1>
void objects_recolor_flu(Objects*, PFarray *flu); // <2>
double objects_mbr_tot_volume(const Objects*);    // <3>

void objects_check_size(const Objects*);
void objects_check_vel(const Objects*, const Dbg*, float dt);
void objects_check_forces(const Objects*, const Dbg*, float dt);

/* membrane distribution */
struct MbrDistr {
    DRbcPack *p;
    DRbcComm *c;
    DRbcUnpack *u;
};

/* rigid distribution */
struct RigDistr {
    DRigPack *p;
    DRigComm *c;
    DRigUnpack *u;
};

/* mesh exchanger */
struct MeshExch {
    EMeshPack *p;
    EMeshComm *c;
    EMeshUnpack *u;
};

/* mesh momentum exchanger */
struct MeshMomExch {
    EMeshPackM *p;
    EMeshCommM *c;
    EMeshUnpackM *u;
};

/* workspace for solvent coloring */
struct Colorer {
    Particle *pp_mesh; /* particle workspace */
    float3 *lo, *hi;   /* bounding boxes     */
};

/* workspace for mesh bounce back */
struct BounceBackData {
    Positioncp *rr_cp;
    MeshMomExch *e;
    Momentum *mm;
};

/* common parts for membranes and rigid objects */
struct Obj {
    char name[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    float mass;           /* mass of one particle                      */
    MeshRead   *mesh;     /* cell template                             */
    MeshWrite  *mesh_write;
    MeshExch   *mesh_exch;
    BounceBackData *bbdata;

    PairParams *fsi;
    PairParams *adhesion;
    WallRepulsePrm *wall_rep_prm;
};

/* data holder for cell membranes */
struct Mbr : Obj {
    RbcQuants q;
    MbrDistr d;
    Force *ff;            /* large time scale forces                   */
    Force *ff_fast;       /* small time scale forces                   */
    RbcForce   *force;    /* helper to compute membrane forces         */
    RbcParams  *params;   /* model parameters                          */
    RbcCom     *com;      /* helper to compute center of masses        */
    RbcStretch *stretch;  /* helper to apply stretching force to cells */
    Triangles  *tri;      /* triangles for one cell on devices         */

    bool active_bounce;
    Colorer  *colorer;
};

/* data holder for rigid objects */
struct Rig : Obj {
    RigQuants q;
    Force *ff;

    RigPinInfo *pininfo;
    RigDistr d;
};

struct Dump {
    Particle *pp; /* workspace on host */
    long id, id_diag;
};

struct Objects {
    int nmbr, nrig;
    Mbr **mbr;
    Rig **rig;
    Opt opt;
    Dump *dump;
    Coords *coords;
    MPI_Comm cart;
    MeshBB *bb;
    bool active; /* false before completely generated */
    PairParams **cnt_params;
};
static void gen_name_mesh_dir(const char *name, char *mesh_dir) {
    strcpy(mesh_dir, "ply/");
    strcat(mesh_dir, name);
}

static void ini_mbr_distr(bool ids, int nv, MPI_Comm comm, int3 L, /**/ MbrDistr *d) {
    UC(drbc_pack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->p));
    UC(drbc_comm_ini(comm, /**/ &d->c));
    UC(drbc_unpack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, int3 L, /**/ RigDistr *d) {
    UC(drig_pack_ini(L, MAX_SOLIDS, nv, /**/ &d->p));
    UC(drig_comm_ini(comm, /**/ &d->c));
    UC(drig_unpack_ini(L,MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_mesh_exch(int3 L, int nv, int max_m, MPI_Comm comm, /**/ MeshExch **me) {
    MeshExch *e;
    EMALLOC(1, me);
    e = *me;
    UC(emesh_pack_ini(L, nv, max_m, /**/ &e->p));
    UC(emesh_comm_ini(comm, /**/ &e->c));
    UC(emesh_unpack_ini(L, nv, max_m, /**/ &e->u));
}

static void ini_mesh_mom_exch(int nt, int max_m, MPI_Comm comm, /**/ MeshMomExch **mme) {
    MeshMomExch *e;
    EMALLOC(1, mme);
    e = *mme;
    UC(emesh_packm_ini(nt, max_m, /**/ &e->p));
    UC(emesh_commm_ini(comm, /**/ &e->c));
    UC(emesh_unpackm_ini(nt, max_m, /**/ &e->u));
}

static void ini_bbdata(int nt, int nv, int max_m, MPI_Comm cart, /**/ BounceBackData **bbdata) {
    BounceBackData *bb;
    EMALLOC(1, bbdata);
    bb = *bbdata;
    UC(ini_mesh_mom_exch(nt, max_m, cart, &bb->e));
    Dalloc(&bb->mm,    max_m * nt);
    Dalloc(&bb->rr_cp, max_m * nv);
}

static void ini_colorer(int nv, int max_m, /**/ Colorer **col) {
    Colorer *c;
    EMALLOC(1, col);
    c = *col;
    Dalloc(&c->pp_mesh, nv * max_m);
    Dalloc(&c->lo, max_m);
    Dalloc(&c->hi, max_m);
}

static void read_params(const Config *cfg, const char *ns, PairParams **par) {
    PairParams *p;
    UC(pair_ini(par));
    p = *par;
    UC(pair_set_conf(cfg, ns, p));
}

static void ini_params(const Config *cfg, const char *name, const char *pair, PairParams **par) {
    const char *ns;
    UC(conf_lookup_string_ns(cfg, name, pair, &ns));

    if (same_str(ns, "none")) *par = NULL;
    else UC(read_params(cfg, ns, par));
}

static void ini_repulsion_params(const Config *cfg, const char *name, WallRepulsePrm **par) {
    const char *ns;
    UC(conf_lookup_string_ns(cfg, name, "repulsion", &ns));

    if (same_str(ns, "none")) *par = NULL;
    else UC(wall_repulse_prm_ini_conf(cfg, ns, par));
}

static void read_mesh(const char *filename, MeshRead **mesh) {
    const char *ext;
    ext = get_filename_ext(filename);

    if      (same_str(ext, "off"))
        UC(mesh_read_ini_off(filename, mesh));
    else if (same_str(ext, "ply"))
        UC(mesh_read_ini_ply(filename, mesh));
    else
        ERR("%s : unrecognised extension <%s>\n", filename, ext);
}

static void ini_obj_common(const Config *cfg, const OptObj *opt, MPI_Comm cart, int max_m, int3 L, Obj *obj) {
    char mesh_dir[FILENAME_MAX];
    int nt, nv;

    strcpy(obj->name, opt->name);
    strcpy(obj->ic_file, opt->ic_file);
    gen_name_mesh_dir(obj->name, mesh_dir);

    UC(read_mesh(opt->templ_file, &obj->mesh));
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, obj->mesh, mesh_dir, /**/ &obj->mesh_write));

    obj->mass = opt->mass;

    UC(ini_params(cfg, obj->name, "fsi", &obj->fsi));
    UC(ini_params(cfg, obj->name, "adhesion", &obj->adhesion));
    UC(ini_repulsion_params(cfg, obj->name, &obj->wall_rep_prm));

    nt = mesh_read_get_nt(obj->mesh);
    nv = mesh_read_get_nv(obj->mesh);

    obj->mesh_exch = NULL;
    obj->bbdata    = NULL;

    if (opt->bounce) {
        UC(ini_bbdata(nt, nv, max_m, cart, /**/ &obj->bbdata));
        UC(ini_mesh_exch(L, nv, max_m, cart, /**/ &obj->mesh_exch));
    }
}

static void ini_mbr(const Config *cfg, const OptMbr *opt, MPI_Comm cart, int3 L,
                    bool recolor, /**/ Mbr **membrane) {
    int nv, max_m;
    Mbr *m;
    EMALLOC(1, membrane);
    m = *membrane;

    max_m = MAX_CELL_NUM;

    m->com       = NULL;
    m->stretch   = NULL;
    m->colorer   = NULL;
    m->active_bounce = false;

    UC(ini_obj_common(cfg, opt, cart, max_m, L, m));

    nv = mesh_read_get_nv(m->mesh);

    Dalloc(&m->ff,      max_m * nv);
    Dalloc(&m->ff_fast, max_m * nv);

    UC(triangles_ini(m->mesh, /**/ &m->tri));
    UC(rbc_ini(max_m, opt->ids, m->mesh, &m->q));
    UC(ini_mbr_distr(opt->ids, nv, cart, L, /**/ &m->d));

    if (opt->dump_com) UC(rbc_com_ini(nv, max_m, /**/ &m->com));
    if (opt->stretch)  UC(rbc_stretch_ini(opt->stretch_file, nv, /**/ &m->stretch));

    UC(rbc_params_ini(&m->params));
    UC(rbc_params_set_conf(cfg, m->name, m->params));

    UC(rbc_force_ini(m->mesh, /**/ &m->force));
    UC(rbc_force_set_conf(m->mesh, cfg, m->name, m->force));

    if (recolor) {
        if (!m->mesh_exch)
            UC(ini_mesh_exch(L, nv, max_m, cart, /**/ &m->mesh_exch));
        UC(ini_colorer(nv, max_m, /**/ &m->colorer));
    }
}

static void ini_rig(const Config *cfg, const OptRig *opt, MPI_Comm cart, int maxp, int3 L, /**/ Rig **rigid) {
    Rig *r;
    long max_m = MAX_SOLIDS;
    int nv;
    EMALLOC(1, rigid);
    r = *rigid;

    UC(ini_obj_common(cfg, opt, cart, max_m, L, r));

    UC(rig_ini(max_m, maxp, r->mesh, &r->q));

    Dalloc(&r->ff, maxp);

    nv = r->q.nv;

    UC(ini_rig_distr(nv, cart, L, /**/ &r->d));

    UC(rig_pininfo_ini(&r->pininfo));
    UC(rig_pininfo_set_conf(cfg, r->name, r->pininfo));
}

static void ini_dump(long maxp, Dump **dump) {
    Dump *d;
    EMALLOC(1, dump);
    d = *dump;
    EMALLOC(maxp, &d->pp);
    d->id = d->id_diag = 0;
}

static bool need_bb(int nm, int nr, const OptMbr *m, const OptRig *r) {
    int i;
    for (i = 0; i < nm; ++i) if (m[i].bounce) return true;
    for (i = 0; i < nr; ++i) if (r[i].bounce) return true;
    return false;
}

static void get_all_names(int nmbr, int nrig, Mbr **mm, Rig **rr, const char **names) {
    int i, j = 0;
    for (i = 0; i < nmbr; ++i) names[j++] = mm[i]->name;
    for (i = 0; i < nrig; ++i) names[j++] = rr[i]->name;
}

enum {NOT_FOUND=-1};

static int get_name_id(const char *s, int n, const char **names) {
    int i;
    for (i = 0; i < n; ++i)
        if (same_str(s, names[i]))
            return i;
    ERR("object <%s> does not exist\n", s);
    return NOT_FOUND;
}

static int get_id_inter(int a, int b) {
    int i, j;
    i = a > b ? a : b;
    j = a > b ? b : a;
    return j + i*(i+1)/2;
}

static void ini_cnt_params(const Config *cfg, int nmbr, int nrig, Mbr **mm, Rig **rr, PairParams ***prms_ptr) {
    int i, n, ninter, ninter_read;
    int id_obja, id_objb, id_inter;
    const char **names, **prms_ns, *sobja, *sobjb, *pair_ns;
    PairParams **prms;

    n = nmbr + nrig;
    ninter = n * (n+1) / 2;

    EMALLOC(ninter, &prms_ns);
    EMALLOC(ninter, prms_ptr);
    EMALLOC(n, &names);

    prms = *prms_ptr;
    get_all_names(nmbr, nrig, mm, rr, names);

    UC(conf_lookup_vstring(cfg, "contact", ninter, &ninter_read, prms_ns));

    for (i = 0; i < ninter; ++i) prms[i] = NULL;

    for (i = 0; i < ninter_read; ++i) {
        UC(conf_lookup_string_ns(cfg, prms_ns[i], "obja", &sobja));
        UC(conf_lookup_string_ns(cfg, prms_ns[i], "objb", &sobjb));
        UC(conf_lookup_string_ns(cfg, prms_ns[i], "pair", &pair_ns));

        id_obja = get_name_id(sobja, n, names);
        id_objb = get_name_id(sobjb, n, names);


        id_inter = get_id_inter(id_obja, id_objb);

        read_params(cfg, pair_ns, &prms[id_inter]);
    }

    EFREE(prms_ns);
    EFREE(names);
}

void objects_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, const Coords *coords, int maxp, Objects **objects) {
    Objects *obj;
    int3 L;
    bool recolor = opt->flu.colors;
    int i;
    EMALLOC(1, objects);
    obj = *objects;
    obj->opt = *opt;
    obj->active = false;

    MC(m::Comm_dup(cart, &obj->cart));
    L = subdomain(coords);
    UC(coords_ini(cart, L.x, L.y, L.z, &obj->coords));

    obj->bb  = NULL;

    obj->nmbr = opt->nmbr;
    obj->nrig = opt->nrig;

    if (obj->nmbr) EMALLOC(obj->nmbr, &obj->mbr);
    if (obj->nrig) EMALLOC(obj->nrig, &obj->rig);

    for (i = 0; i < obj->nmbr; ++i) UC(ini_mbr(cfg, &opt->mbr[i], cart, L, recolor, &obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(ini_rig(cfg, &opt->rig[i], cart, maxp, L, &obj->rig[i]));

    if (need_bb(opt->nmbr, opt->nrig, opt->mbr, opt->rig))
        UC(mesh_bounce_ini(maxp, /**/ &obj->bb));

    UC(ini_dump(maxp, &obj->dump));
    UC(ini_cnt_params(cfg, obj->nmbr, obj->nrig, obj->mbr, obj->rig, &obj->cnt_params));
}
static void fin_mbr_distr(/**/ MbrDistr *d) {
    UC(drbc_pack_fin(/**/ d->p));
    UC(drbc_comm_fin(/**/ d->c));
    UC(drbc_unpack_fin(/**/ d->u));
}

static void fin_rig_distr(/**/ RigDistr *d) {
    UC(drig_pack_fin(/**/ d->p));
    UC(drig_comm_fin(/**/ d->c));
    UC(drig_unpack_fin(/**/ d->u));
}

static void fin_mesh_exch(/**/ MeshExch *e) {
    UC(emesh_pack_fin(   /**/ e->p));
    UC(emesh_comm_fin(   /**/ e->c));
    UC(emesh_unpack_fin( /**/ e->u));
    EFREE(e);
}

static void fin_mesh_mom_exch(/**/ MeshMomExch *e) {
    UC(emesh_packm_fin(   /**/ e->p));
    UC(emesh_commm_fin(   /**/ e->c));
    UC(emesh_unpackm_fin( /**/ e->u));
    EFREE(e);
}

static void fin_bbdata(/**/ BounceBackData *bb) {
    UC(fin_mesh_mom_exch(/**/ bb->e));
    Dfree(bb->mm);
    Dfree(bb->rr_cp);
    EFREE(bb);
}

static void fin_colorer(/**/ Colorer *c) {
    Dfree(c->pp_mesh);
    Dfree(c->lo);
    Dfree(c->hi);
    EFREE(c);
}

static void fin_obj_common(Obj *o) {
    UC(mesh_read_fin(o->mesh));
    UC(mesh_write_fin(o->mesh_write));

    if (o->fsi)      UC(pair_fin(o->fsi));
    if (o->adhesion) UC(pair_fin(o->adhesion));
    if (o->wall_rep_prm)
        UC(wall_repulse_prm_fin(o->wall_rep_prm));
}

static void fin_mbr(Mbr *m) {
    UC(rbc_fin(&m->q));
    UC(rbc_force_fin(m->force));

    UC(fin_mbr_distr(/**/ &m->d));

    Dfree(m->ff);
    Dfree(m->ff_fast);
    UC(triangles_fin(m->tri));

    if (m->com)     UC(rbc_com_fin(/**/ m->com));
    if (m->stretch) UC(rbc_stretch_fin(/**/ m->stretch));

    UC(rbc_params_fin(m->params));

    if (m->colorer)   UC(fin_colorer(/**/ m->colorer));
    if (m->mesh_exch) UC(fin_mesh_exch(/**/ m->mesh_exch));

    UC(fin_obj_common(m));

    EFREE(m);
}

static void fin_rig(Rig *r) {
    UC(rig_fin(&r->q));
    Dfree(r->ff);

    UC(fin_rig_distr(/**/ &r->d));
    UC(rig_pininfo_fin(r->pininfo));

    if (r->bbdata) UC(fin_bbdata(r->bbdata));

    UC(fin_obj_common(r));

    EFREE(r);
}

static void fin_dump(Dump *d) {
    EFREE(d->pp);
    EFREE(d);
}

static void fin_cnt_params(int nmbr, int nrig, PairParams **prms) {
    int i, n, ninter;
    n = nmbr + nrig;
    ninter = n * (n+1) / 2;
    for (i = 0; i < ninter; ++i) {
        if (prms[i])
            UC(pair_fin(prms[i]));
    }
    EFREE(prms);
}

void objects_fin(Objects *obj) {
    int i;
    for (i = 0; i < obj->nmbr; ++i) UC(fin_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(fin_rig(obj->rig[i]));
    if (obj->nmbr) EFREE(obj->mbr);
    if (obj->nrig) EFREE(obj->rig);
    if (obj->bb)  UC(mesh_bounce_fin(/**/ obj->bb));
    UC(fin_cnt_params(obj->nmbr, obj->nrig, obj->cnt_params));
    UC(fin_dump(obj->dump));
    UC(coords_fin(obj->coords));
    MC(m::Comm_free(&obj->cart));
    EFREE(obj);
}
static void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

static void check_size_mbr(const Mbr *m) {
    UC(check_size(m->q.nc, MAX_CELL_NUM));
}

static void check_size_rig(const Rig *r) {
    UC(check_size(r->q.ns, MAX_SOLIDS));
}

void objects_check_size(const Objects *obj) {
    int i;
    for (i = 0; i < obj->nmbr; ++i) UC(check_size_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(check_size_rig(obj->rig[i]));
}

static void check_vel_mbr(const Dbg *dbg, float dt, const Coords *coords, const Mbr *m) {
    UC(dbg_check_vel(dt, coords, "rbc", dbg, m->q.n, m->q.pp));
}

static void check_vel_rig(const Dbg *dbg, float dt, const Coords *coords, const Rig *r) {
    UC(dbg_check_vel(dt, coords, "rig", dbg, r->q.n, r->q.pp));
}

void objects_check_vel(const Objects *o, const Dbg *dbg, float dt) {
    int i;
    for (i = 0; i < o->nmbr; ++i) UC(check_vel_mbr(dbg, dt, o->coords, o->mbr[i]));
    for (i = 0; i < o->nrig; ++i) UC(check_vel_rig(dbg, dt, o->coords, o->rig[i]));
}

static void check_forces_mbr(const Dbg *dbg, float dt, const Coords *coords, const Mbr *m) {
    UC(dbg_check_forces(dt, coords, "rbc.ff", dbg, m->q.n, m->q.pp, m->ff));
}

static void check_forces_rig(const Dbg *dbg, float dt, const Coords *coords, const Rig *r) {
    UC(dbg_check_forces(dt, coords, "rig.ff", dbg, r->q.n, r->q.pp, r->ff));
}

void objects_check_forces(const Objects *o, const Dbg *dbg, float dt) {
    int i;
    for (i = 0; i < o->nmbr; ++i) UC(check_forces_mbr(dbg, dt, o->coords, o->mbr[i]));
    for (i = 0; i < o->nrig; ++i) UC(check_forces_rig(dbg, dt, o->coords, o->rig[i]));
}
static void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

static void clear_mbr_forces(Mbr *m) {
    UC(clear_forces(m->q.n, m->ff));
}

static void clear_rig_forces(Rig *r) {
    UC(clear_forces(r->q.n, r->ff));
    UC(rig_reinit_ft(r->q.ns, /**/ r->q.ss));
}

void objects_clear_forces(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(clear_mbr_forces(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(clear_rig_forces(obj->rig[i]));
}

// used in main.h
static void internal_forces_mbr(float dt, const OptMbr *opt, Mbr *m) {
    UC(rbc_force_apply(m->force, m->params, dt, &m->q, /**/ m->ff_fast));
    if (opt->stretch) UC(rbc_stretch_apply(m->q.nc, m->stretch, /**/ m->ff_fast));
}

static void bforce_mbr(const Coords *c, const BForce *bf, Mbr *m) {
    UC(bforce_apply(c, m->mass, bf, m->q.n, m->q.pp, /**/ m->ff));
}

static void bforce_rig(const Coords *c, const BForce *bf, Rig *r) {
    UC(bforce_apply(c, r->mass, bf, r->q.n, r->q.pp, /**/ r->ff));
}

void objects_body_forces(const BForce *bf, Objects *o) {
    int i;
    const Opt *opt = &o->opt;
    if (!o->active) return;

    for (i = 0; i < o->nmbr; ++i)
        if (opt->mbr[i].push)
            bforce_mbr(o->coords, bf, o->mbr[i]);

    for (i = 0; i < o->nrig; ++i)
        if (opt->rig[i].push)
            bforce_rig(o->coords, bf, o->rig[i]);
}
static void save_mesh_mbr(Mbr *m) {
    if (m->bbdata)
        convert_pp2rr_previous(m->q.n, m->q.pp, m->bbdata->rr_cp);
}

static void save_mesh_rig(Rig *r) {
    if (r->bbdata)
        convert_pp2rr_previous(r->q.ns * r->q.nv, r->q.i_pp, r->bbdata->rr_cp);
}

void objects_save_mesh(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(save_mesh_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(save_mesh_rig(obj->rig[i]));
}

static void clear_vel_mbr(Mbr *m) {
    UC(scheme_move_clear_vel(m->q.n, m->q.pp));
}

static void clear_vel_rig(Rig *r) {
    UC(scheme_move_clear_vel(r->q.n, r->q.pp));
}

void objects_clear_vel(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(clear_vel_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(clear_vel_rig(obj->rig[i]));
}

static void advance_rig(float dt, Rig *r) {
    if (!r->q.n) return;
    RigQuants *q = &r->q;
    UC(rig_update(r->pininfo, dt, q->n, r->ff, q->rr0, q->ns, /**/ q->pp, q->ss));
    UC(rig_update_mesh(dt, q->ns, q->ss, q->nv, q->dvv, /**/ q->i_pp));
}

static void ini_mbr_fast_forces(Mbr *m) {
    aD2D(m->ff_fast, m->ff, m->q.n);
}

static void sub_advance_mbr(float dt, const OptMbr *opt, Mbr *m) {
    UC(ini_mbr_fast_forces(m));
    UC(internal_forces_mbr(dt, opt, m));
    UC(scheme_move_apply(dt, m->mass, m->q.n, m->ff_fast, m->q.pp));
}

static void advance_mbr(float dt, const OptMbr *opt, Mbr *m) {
    int i, n = opt->substeps;
    float sub_dt = dt / n;
    for (i = 0; i < n; ++i)
        UC(sub_advance_mbr(sub_dt, opt, m));
}

void objects_advance(float dt, Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(advance_mbr(dt, &obj->opt.mbr[i], obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(advance_rig(dt, obj->rig[i]));
}

static void distribute_mbr(Mbr *m) {
    RbcQuants *q = &m->q;
    MbrDistr  *d = &m->d;

    UC(drbc_build_map(q->nc, q->nv, q->pp, /**/ d->p));
    UC(drbc_pack(q, /**/ d->p));
    UC(drbc_download(/**/d->p));

    UC(drbc_post_send(d->p, d->c));
    UC(drbc_post_recv(d->c, d->u));

    UC(drbc_unpack_bulk(d->p, /**/ q));

    UC(drbc_wait_send(d->c));
    UC(drbc_wait_recv(d->c, d->u));

    UC(drbc_unpack_halo(d->u, /**/ q));
    dSync();
}

static void distribute_rig(Rig *r) {
    RigQuants *q = &r->q;
    RigDistr  *d = &r->d;
    int nv = q->nv;

    UC(drig_build_map(q->ns, q->ss, /**/ d->p));
    UC(drig_pack(q->ns, nv, q->ss, q->i_pp, /**/ d->p));
    UC(drig_download(/**/d->p));

    UC(drig_post_send(d->p, d->c));
    UC(drig_post_recv(d->c, d->u));

    UC(drig_unpack_bulk(d->p, /**/ q));

    UC(drig_wait_send(d->c));
    UC(drig_wait_recv(d->c, d->u));

    UC(drig_unpack_halo(d->u, /**/ q));

    q->n = q->ns * q->nps;
    UC(rig_generate(q->ns, q->ss, q->nps, q->rr0, /**/ q->pp));
    dSync();
}

void objects_distribute (Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(distribute_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(distribute_rig(obj->rig[i]));
}

static void update_dpd_prms(float dt, float kBT, Obj *o) {
    if (o->fsi) UC(pair_compute_dpd_sigma(kBT, dt, /**/ o->fsi));
}

void objects_update_dpd_prms(float dt, float kBT, Objects *obj) {
    int i, n, ninter;
    for (i = 0; i < obj->nmbr; ++i) update_dpd_prms(dt, kBT, obj->mbr[i]);
    for (i = 0; i < obj->nrig; ++i) update_dpd_prms(dt, kBT, obj->rig[i]);

    n = obj->nmbr + obj->nrig;
    ninter = n*(n+1)/2;
    for (i = 0; i < ninter; ++i) {
        if (obj->cnt_params[i])
            UC(pair_compute_dpd_sigma(kBT, dt, /**/ obj->cnt_params[i]));
    }
}

bool objects_have_bounce(const Objects *o) {
    return o->bb != NULL;
}

static void get_mbr(Mbr *m, PFarrays *pf) {
    PaArray p;
    FoArray f;
    parray_push_pp(m->q.pp, &p);
    farray_push_ff(m->ff, &f);
    UC(pfarrays_push(pf, m->q.n, p, f));
}

static void get_rig(Rig *r, PFarrays *pf) {
    PaArray p;
    FoArray f;
    parray_push_pp(r->q.pp, &p);
    farray_push_ff(r->ff, &f);
    UC(pfarrays_push(pf, r->q.n, p, f));
}

void objects_get_particles_all(Objects *obj, PFarrays *pf) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr(obj->mbr[i], pf);
    for (i = 0; i < obj->nrig; ++i) get_rig(obj->rig[i], pf);
}

void objects_get_particles_mbr(Objects *obj, PFarrays *pf) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr(obj->mbr[i], pf);
}

static void get_mbr_accel(const Mbr *m, TimeStepAccel *aa) {
    if(m->q.n) UC(time_step_accel_push(aa, m->mass, m->q.n, m->ff));
}

static void get_rig_accel(const Rig *r, TimeStepAccel *aa) {
    if (r->q.n) UC(time_step_accel_push(aa, r->mass, r->q.n, r->ff));
}

void objects_get_accel(const Objects *obj, TimeStepAccel *aa) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr_accel(obj->mbr[i], aa);
    for (i = 0; i < obj->nrig; ++i) get_rig_accel(obj->rig[i], aa);
}

static void get_params_fsi(const Obj *o, const PairParams **prm) {*prm = o->fsi;}

void objects_get_params_fsi(const Objects *obj, const PairParams *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_fsi(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_fsi(obj->rig[i], &prms[j++]);
}

static void get_params_adhesion(const Obj *o, const PairParams **prm) {*prm = o->adhesion;}

void objects_get_params_adhesion(const Objects *obj, const PairParams *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_adhesion(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_adhesion(obj->rig[i], &prms[j++]);
}

static void get_params_repulsion(const Obj *o, const WallRepulsePrm **prm) {*prm = o->wall_rep_prm;}

void objects_get_params_repulsion(const Objects *obj, const WallRepulsePrm *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_repulsion(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_repulsion(obj->rig[i], &prms[j++]);
}

void objects_get_params_cnt(const Objects *o, const PairParams ***prms) {
    *prms = (const PairParams**) o->cnt_params;
}

static void restart_mbr(MPI_Comm cart, const char *base, Mbr *m) {
    UC(rbc_strt_quants(cart, m->mesh, base, m->name, RESTART_BEGIN, &m->q));
}

static void restart_rig(MPI_Comm cart, const char *base, Rig *r) {
    UC(rig_strt_quants(cart, r->mesh, base, r->name, RESTART_BEGIN, &r->q));
}

void objects_restart(Objects *o) {
    int i;
    const char *base = o->opt.dump.strt_base_read;
    for (i = 0; i < o->nmbr; ++i) restart_mbr(o->cart, base, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) restart_rig(o->cart, base, o->rig[i]);
    o->active = true;
}

static double local_vol_mbr(const Mbr *m) {
    long nc;
    double V0;
    nc = m->q.nc;
    V0 = rbc_params_get_tot_volume(m->params);
    return nc * V0;
}

double objects_mbr_tot_volume(const Objects *o) {
    int i;
    double loc, tot;
    if (!o->active) return 0;

    loc = 0;

    for (i = 0; i < o->nmbr; ++i)
        loc += local_vol_mbr(o->mbr[i]);

    tot = 0;
    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, o->cart));

    return tot;
}

static void activate_mbr_bounce(Mbr *m) {
    if (m->bbdata)
        m->active_bounce = true;
}

void objects_activate_mbr_bounce(Objects *o) {
    int i;
    for (i = 0; i < o->nmbr; ++i) activate_mbr_bounce(o->mbr[i]);
}
static MeshInfo mesh_info_mbr(const Mbr *m) {
    MeshInfo mi;
    mi.nv = m->q.nv;
    mi.nt = m->q.nt;
    mi.tt = m->tri->tt;
    return mi;
}

static MeshInfo mesh_info_rig(const Rig *r) {
    MeshInfo mi;
    mi.nv = r->q.nv;
    mi.nt = r->q.nt;
    mi.tt = r->q.dtt;
    return mi;
}

static void mesh_pack_and_send_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    MeshExch *e = m->mesh_exch;

    UC(emesh_build_map(q->nc, q->nv, q->pp, /**/ e->p));
    UC(emesh_pack_rrcp(q->nv, m->bbdata->rr_cp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));
}

static void mesh_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    MeshExch  *e = r->mesh_exch;

    UC(emesh_build_map(q->ns, q->nv, q->i_pp, /**/ e->p));
    UC(emesh_pack_rrcp(q->nv, r->bbdata->rr_cp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));
}

static int mesh_recv_unpack_mbr(Mbr *m) {
    MeshExch *e = m->mesh_exch;
    RbcQuants *q = &m->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    /* unpack at the end of current mesh buffer */
    UC(emesh_unpack_rrcp(q->nv, e->u, /**/ &nmhalo, m->bbdata->rr_cp + q->n));
    return nmhalo;
}

static int mesh_recv_unpack_rig(Rig *r) {
    MeshExch *e = r->mesh_exch;
    RigQuants *q = &r->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    /* unpack at the end of current mesh buffer */
    UC(emesh_unpack_rrcp(q->nv, e->u, /**/ &nmhalo, r->bbdata->rr_cp + q->ns * q->nv));
    return nmhalo;
}

static void clear_momentum(long n, Momentum *mm) {
    if (n) DzeroA(mm, n);
}

static void clear_momentum_mbr(int nmhalo, Mbr *m) {
    long ntri = m->q.nt * (m->q.nc + nmhalo);
    UC(clear_momentum(ntri, m->bbdata->mm));
}

static void clear_momentum_rig(int nmhalo, Rig *r) {
    long ntri = r->q.nt * (r->q.ns + nmhalo);
    UC(clear_momentum(ntri, r->bbdata->mm));
}


static void find_and_select_collisions_rig(float dt, long nflu, const Clist *cflu, const Particle *flu_pp, const Particle *flu_pp0, int nm, Rig *r, MeshBB *bb) {
    MeshInfo mi = mesh_info_rig(r);
    UC(mesh_bounce_find_collisions(dt, nm, mi, r->bbdata->rr_cp,
                                   clists_get_dim(cflu),
                                   clists_get_ss(cflu),
                                   clists_get_cc(cflu),
                                   flu_pp, flu_pp0, /**/ bb));
    UC(mesh_bounce_select_collisions(nflu, /**/ bb));
}

static void find_and_select_collisions_mbr(float dt, long nflu, const Clist *cflu, const Particle *flu_pp, const Particle *flu_pp0, int nm, Mbr *m, MeshBB *bb) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_find_collisions(dt, nm, mi, m->bbdata->rr_cp,
                                   clists_get_dim(cflu),
                                   clists_get_ss(cflu),
                                   clists_get_cc(cflu),
                                   flu_pp, flu_pp0, /**/ bb));
    UC(mesh_bounce_select_collisions(nflu, /**/ bb));
}

static void bounce_mbr(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Mbr *m) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, m->bbdata->rr_cp, pp0, /**/ pp, m->bbdata->mm));
}

static void bounce_rig(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Rig *r) {
    MeshInfo mi = mesh_info_rig(r);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, r->bbdata->rr_cp, pp0, /**/ pp, r->bbdata->mm));
}

static void mom_pack_and_send_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshMomExch *e = bb->e;
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(m->mesh_exch->u, /**/ counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->n, /**/ e->p));
    UC(emesh_downloadM(counts, /**/ e->p));

    UC(emesh_post_recv(e->c, e->u));
    UC(emesh_post_send(e->p, e->c));
}

static void mom_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(r->mesh_exch->u, /**/ counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->ns * q->nt, /**/ e->p));
    UC(emesh_downloadM(counts, /**/ e->p));

    UC(emesh_post_recv(e->c, e->u));
    UC(emesh_post_send(e->p, e->c));
}

static void mom_recv_unpack_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshMomExch *e = bb->e;
    MeshExch *meshe = m->mesh_exch;

    UC(emesh_wait_recv(e->c, e->u));
    UC(emesh_wait_send(e->c));

    UC(emesh_upload(e->u));
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u, /**/ bb->mm));
}

static void mom_recv_unpack_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;
    MeshExch *meshe = r->mesh_exch;

    UC(emesh_wait_recv(e->c, e->u));
    UC(emesh_wait_send(e->c));

    UC(emesh_upload(e->u));
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u, /**/ bb->mm));
}

static void collect_mom_mbr(float dt, Mbr *m) {
    RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshInfo mi = mesh_info_mbr(m);

    UC(mesh_bounce_collect_rbc_momentum(m->mass, q->nc, mi, bb->mm, /**/ q->pp));
}

static void collect_mom_rig(Rig *r) {
    RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshInfo mi = mesh_info_rig(r);

    UC(mesh_bounce_collect_rig_momentum(q->ns, mi, q->i_pp, bb->mm, /**/ q->ss));
}

/* TODO less brutal implementation */
static void bounce_mbr(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Mbr *m) {
    int nmhalo;
    if (!m->bbdata || !m->active_bounce) return;

    mesh_pack_and_send_mbr(m);
    nmhalo = mesh_recv_unpack_mbr(m);

    UC(mesh_bounce_reini(n, /**/ bb));
    clear_momentum_mbr(nmhalo, m);

    find_and_select_collisions_mbr(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + m->q.nc, m, bb);

    bounce_mbr(dt, flu_mass, bb, n, flu_pp0, /**/ flu_pp, m);

    mom_pack_and_send_mbr(m);
    mom_recv_unpack_mbr(m);
    collect_mom_mbr(dt, m);
}

static void bounce_rig(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Rig *r) {
    int nmhalo;
    if (!r->bbdata) return;

    mesh_pack_and_send_rig(r);
    nmhalo = mesh_recv_unpack_rig(r);

    UC(mesh_bounce_reini(n, /**/ bb));
    clear_momentum_rig(nmhalo, r);

    find_and_select_collisions_rig(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + r->q.ns, r, bb);

    bounce_rig(dt, flu_mass, bb, n, flu_pp0, /**/ flu_pp, r);

    mom_pack_and_send_rig(r);
    mom_recv_unpack_rig(r);
    collect_mom_rig(r);
}

static void save_mesh_mbr_current(Mbr *m) {
    if (m->bbdata && m->active_bounce)
        convert_pp2rr_current(m->q.n, m->q.pp, m->bbdata->rr_cp);
}

static void save_mesh_rig_current(Rig *r) {
    if (r->bbdata)
        convert_pp2rr_current(r->q.ns * r->q.nv, r->q.i_pp, r->bbdata->rr_cp);
}

static void objects_save_mesh_current(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(save_mesh_mbr_current(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(save_mesh_rig_current(obj->rig[i]));
}

void objects_bounce(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, Objects *obj) {
    MeshBB *bb = obj->bb;
    int i;

    if (!obj->active) return;
    if (!bb) return;

    UC(objects_save_mesh_current(obj));

    for (i = 0; i < obj->nmbr; ++i)
        bounce_mbr(dt, flu_mass, flu_cells, n, flu_pp0, flu_pp, bb, obj->mbr[i]);

    for (i = 0; i < obj->nrig; ++i)
        bounce_rig(dt, flu_mass, flu_cells, n, flu_pp0, flu_pp, bb, obj->rig[i]);
}
static void reini_colors(PFarray *flu) {
    UC(collision_label_ini(flu->n, BLUE_COLOR, /**/ (int*) flu->p.cc));
}

static int exchange_mesh_mbr(Mbr *m, Particle *pp) {
    int nm, nv, nmhalo;
    const RbcQuants *q = &m->q;
    MeshExch *e = m->mesh_exch;
    nm = q->nc;
    nv = q->nv;

    UC(emesh_build_map(nm, nv, q->pp, /**/ e->p));
    UC(emesh_pack_pp(nv, q->pp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));

    if (nm*nv) aD2D(pp, q->pp, nm*nv);

    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    UC(emesh_unpack_pp(nv, e->u, /**/ &nmhalo, pp + nm * nv));
    return nm + nmhalo;
}

static void recolor_flu_from_mbr(Mbr *m, PFarray *flu) {
    int nm, nv;
    Colorer *c = m->colorer;
    if (c == NULL)
        ERR("colorer was not initialised");

    nv = m->q.nv;
    nm = exchange_mesh_mbr(m, c->pp_mesh);

    UC(minmax(c->pp_mesh, nv, nm, /**/ c->lo, c->hi));

    UC(collision_label(NOT_PERIODIC, flu->n, (const Particle*) flu->p.pp, m->tri, nv, nm,
                       c->pp_mesh, c->lo, c->hi, RED_COLOR, /**/ (int*) flu->p.cc));
}

void objects_recolor_flu(Objects *obj, PFarray *flu) {
    int i;
    if (!obj->active) return;
    if (!obj->nmbr) return;
    if (flu->n == 0) return;

    UC(reini_colors(flu));
    for (i = 0; i < obj->nmbr; ++i) UC(recolor_flu_from_mbr(obj->mbr[i], flu));
}
static void dump_mesh_mbr(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Mbr *m) {
    cD2H(pp, m->q.pp, m->q.n);
    UC(mesh_write_particles(m->mesh_write, cart, coords, m->q.nc, pp, id));
}

static void dump_mesh_rig(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Rig *r) {
    cD2H(pp, r->q.i_pp, r->q.ns * r->q.nv);
    UC(mesh_write_particles(r->mesh_write, cart, coords, r->q.ns, pp, id));
}

void objects_mesh_dump(Objects *obj) {
    int i;
    Dump *d = obj->dump;
    if (!obj->active) return;

    for (i = 0; i < obj->nmbr; ++i)
        UC(dump_mesh_mbr(obj->cart, obj->coords, d->pp, d->id, obj->mbr[i]));

    for (i = 0; i < obj->nrig; ++i)
        UC(dump_mesh_rig(obj->cart, obj->coords, d->pp, d->id, obj->rig[i]));
    ++ d->id;
}

static void dump_diag_mbr(MPI_Comm cart, const Coords *coords, Mbr *m, Dump *d) {
    RbcQuants *q = &m->q;
    long nc = q->nc;
    float3 *rr, *vv;
    if (m->com) {
        UC(rbc_com_apply(m->com, nc, q->pp, /**/ &rr, &vv));
        UC(io_com_dump(cart, coords, m->name, d->id_diag, nc, q->ii, rr));
    }
}

static void dump_diag_rig(MPI_Comm cart, const Coords *coords, Rig *r, Dump *d) {
    RigQuants *q = &r->q;
    cD2H(q->ss_dmp, q->ss, q->ns);
    UC(io_rig_dump(cart, coords, r->name, d->id_diag, q->ns, q->ss_dmp));
}

void objects_diag_dump(float t, Objects *obj) {
    int i;
    Dump *d = obj->dump;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) dump_diag_mbr(obj->cart, obj->coords, obj->mbr[i], d);
    for (i = 0; i < obj->nrig; ++i) dump_diag_rig(obj->cart, obj->coords, obj->rig[i], d);
    ++ d->id_diag;
}

static void dump_part_rig(MPI_Comm cart, const Coords *coords, long id, Rig *r, IoBop *bop) {
    RigQuants *q = &r->q;
    cD2H(q->pp_hst, q->pp, q->n);
    UC(io_bop_parts(cart, coords, q->n, q->pp_hst, r->name, id, bop));
}

void objects_part_dump(long id, Objects *o, IoBop *bop) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nrig; ++i) dump_part_rig(o->cart, o->coords, id, o->rig[i], bop);
}

void objects_strt_templ(const char *base, Objects *o) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nrig; ++i) UC(rig_strt_dump_templ(o->cart, base, o->rig[i]->name, &o->rig[i]->q));
}

void objects_strt_dump(const char *base, long id, Objects *o) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nmbr; ++i) UC(rbc_strt_dump(o->cart, base, o->mbr[i]->name, id, &o->mbr[i]->q));
    for (i = 0; i < o->nrig; ++i) UC(rig_strt_dump(o->cart, base, o->rig[i]->name, id, &o->rig[i]->q));
}
static void gen_mesh_mbr(Coords *coords, MPI_Comm cart, Mbr *m) {
    MeshRead *mesh = m->mesh;
    UC(rbc_gen_mesh(coords, cart, mesh, m->ic_file, /**/ &m->q));
}

static void gen_mesh_rig(Coords *coords, MPI_Comm cart, Rig *r) {
    MeshRead *mesh = r->mesh;
    UC(rig_gen_mesh(coords, cart, mesh, r->ic_file, /**/ &r->q));
}

void objects_gen_mesh(Objects *o) {
    int i;
    for (i = 0; i < o->nmbr; ++i) gen_mesh_mbr(o->coords, o->cart, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) gen_mesh_rig(o->coords, o->cart, o->rig[i]);
}

template <typename T>
static void remove(T *data, int nv, int *stay, int nc) {
    int c; /* c: cell index */
    for (c = 0; c < nc; c++)
        cA2A(data + nv*c, data + nv * stay[c], nv);
}

static void remove_mbr(const Sdf *sdf, Mbr *m) {
    int nc0, stay[MAX_CELL_NUM];
    RbcQuants *q = &m->q;
    q->nc = sdf_who_stays(sdf, q->n, q->pp, nc0 = q->nc, q->nv, /**/ stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    msg_print("%s: %d/%d survived", m->name, q->nc, nc0);
}

static void remove_rig(const Sdf *sdf, Rig *r) {
    int nip, ns0, stay[MAX_SOLIDS];
    RigQuants *q = &r->q;
    nip = q->ns * q->nv;
    q->ns = sdf_who_stays(sdf, nip, q->i_pp, ns0 = q->ns, q->nv, /**/ stay);
    q->n  = q->ns * q->nps;
    remove(q->pp,       q->nps,      stay, q->ns);
    remove(q->pp_hst,   q->nps,      stay, q->ns);

    remove(q->ss,       1,           stay, q->ns);
    remove(q->ss_hst,   1,           stay, q->ns);

    remove(q->i_pp,     q->nv, stay, q->ns);
    remove(q->i_pp_hst, q->nv, stay, q->ns);
    msg_print("%s: %d/%d survived", r->name, q->ns, ns0);
}

void objects_remove_from_wall(const Sdf *sdf, Objects *o) {
    int i;
    if (!sdf) return;
    for (i = 0; i < o->nmbr; ++i) remove_mbr(sdf, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) remove_rig(sdf, o->rig[i]);
}

static void gen_freeze_mbr(MPI_Comm cart, Mbr *m) {
    UC(rbc_gen_freeze(cart, /**/ &m->q));
}

static void gen_freeze_rig(const Coords *coords, MPI_Comm cart, bool empty_pp, int numdensity, PFarray *flu, Rig *r) {
    const MeshRead *mesh = r->mesh;
    int n;
    Particle *pp = (Particle*) flu->p.pp;
    n = flu->n;

    UC(rig_gen_freeze(coords, empty_pp, numdensity, r->mass, r->pininfo,
                      cart, mesh, pp, &n, /**/ &r->q));

    flu->n = n;
}

void objects_gen_freeze(PFarray *flu, Objects *o) {
    int i;
    const Opt *opt = &o->opt;

    for (i = 0; i < o->nmbr; ++i)
        gen_freeze_mbr(o->cart, o->mbr[i]);

    for (i = 0; i < o->nrig; ++i)
        gen_freeze_rig(o->coords, o->cart, opt->rig[i].empty_pp, opt->params.numdensity, flu, o->rig[i]);

    o->active = true;
}





#include <stdio.h>
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

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
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
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

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

#ifndef UDX_SIM_OBJECTS_PARRAY_COMMON
#define UDX_SIM_OBJECTS_PARRAY_COMMON
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
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
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
#endif
#ifndef UDX_SIM_OBJECTS_PFARRAYS_COMMON
#define UDX_SIM_OBJECTS_PFARRAYS_COMMON
struct PFarray {
    PaArray p;
    FoArray f;
    long n;
};

struct PFarrays;

struct PaArray;
struct FoArray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarrays_clear(PFarrays*);
void pfarrays_push(PFarrays*, long n, PaArray, FoArray);
void pfarrays_push(PFarrays*, PFarray);
int  pfarrays_size(const PFarrays*);
void pfarrays_get(int i, const PFarrays*, long *n, PaArray*, FoArray*);
void pfarrays_get(int i, const PFarrays*, PFarray*);
#endif

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

struct Contact;
struct int3;
struct PairParams;

void cnt_ini(int maxp, int rank, int3 L, int nobj, /**/ Contact **c);
void cnt_fin(Contact *c);

void cnt_build_cells(int nw, const PairParams **prms, const PaWrap *pw, /**/ Contact *c); // <1>
void cnt_bulk(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw); // <2>
void cnt_halo(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts); // <3>
struct RNDunif;
struct Fsi;
struct int3;

void fsi_ini(int rank, int3 L, /**/ Fsi **fsi);
void fsi_fin(Fsi *fsi);

void fsi_bind_solvent(PaArray pa, Force *ff, int n, const int *starts, /**/ Fsi *fsi);  // <1>
void fsi_bulk(Fsi *fsi, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw); // <2>
void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts); // <3>

struct EObjPack;
struct EObjComm;
struct EObjUnpack;

struct EObjPackF;
struct EObjCommF;
struct EObjUnpackF;

void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **p);
void eobj_comm_ini(MPI_Comm cart, /**/ EObjComm **c);
void eobj_unpack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjUnpack **u);

void eobj_pack_fin(EObjPack *p);
void eobj_comm_fin(EObjComm *c);
void eobj_unpack_fin(EObjUnpack *u);

void eobj_build_map(int nw, const PaWrap *ww, /**/ EObjPack *p);

void eobj_pack(int nw, const PaWrap *ww, /**/ EObjPack *p);
void eobj_download(int nw, EObjPack *p);

void eobj_post_recv(EObjComm *c, EObjUnpack *u);
void eobj_post_send(EObjPack *p, EObjComm *c);
void eobj_wait_recv(EObjComm *c, EObjUnpack *u);
void eobj_wait_send(EObjComm *c);

int26 eobj_get_counts(EObjUnpack *u); // <1>
void  eobj_get_all_counts(int nw, EObjUnpack *u, int *all_counts);
Pap26 eobj_upload_shift(EObjUnpack *u); // <2>
Fop26 eobj_reini_ff(const EObjUnpack *u, EObjPackF *pf); // <3>


void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **p);
void eobj_commf_ini(MPI_Comm cart, /**/ EObjCommF **c);
void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **u);

void eobj_packf_fin(EObjPackF *p);
void eobj_commf_fin(EObjCommF *c);
void eobj_unpackf_fin(EObjUnpackF *u);

void eobj_download_ff(EObjPackF *p);

void eobj_post_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_post_send_ff(EObjPackF *p, EObjCommF *c);
void eobj_wait_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_wait_send_ff(EObjCommF *c);

void eobj_unpack_ff(EObjUnpackF *u, const EObjPack *p, int nw, /**/ FoWrap *ww);

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON
/* solvent options */
struct OptFlu {
    bool colors, ids, ss, push;
};

/* common parameters for membranes and rigid objects */
struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};

/* membrane options */
struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};

/* rigid options */
struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct OptDump {
    bool       parts,      mesh,      field,      strt, forces;
    float freq_parts, freq_mesh, freq_field, freq_strt, freq_diag;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
};

struct Opt {
    OptFlu flu;
    OptMbr mbr[MAX_MBR_TYPES];
    OptRig rig[MAX_RIG_TYPES];
    int nmbr, nrig;
    OptWall wall;
    OptParams params;
    OptDump dump;
    bool inflow, outflow, denoutflow, vcon;
    int recolor_freq;
    int sampler_npdump;
    int3 sampler_grid_ref;
    bool restart;
    bool tracers;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
long opt_estimate_maxp(const Opt*);
#endif

struct ObjInter;

struct Config;
struct Opt;
struct PFarray;
struct PFarrays;

void obj_inter_ini(const Config*, const Opt*, MPI_Comm, float dt, int maxp, /**/ ObjInter**);
void obj_inter_fin(ObjInter*);

void obj_inter_forces(ObjInter*, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj); // <1>
/* object exchanger for forces */
struct ObjExch {
    EObjPack *p;
    EObjComm *c;
    EObjUnpack *u;

    EObjPackF *pf;
    EObjCommF *cf;
    EObjUnpackF *uf;
};

/* helper for computing object interactions */
struct ObjInter {
    ObjExch *e;
    Contact *cnt;
    Fsi     *fsi;
};

static void ini_obj_exch(MPI_Comm comm, int3 L, /**/ ObjExch **oe) {
    ObjExch *e;
    int maxpsolid = MAX_PSOLID_NUM;
    EMALLOC(1, oe);
    e = *oe;

    UC(eobj_pack_ini   (L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->p));
    UC(eobj_comm_ini   (comm,                                         /**/ &e->c));
    UC(eobj_unpack_ini (L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->u));

    UC(eobj_packf_ini  (L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->pf));
    UC(eobj_commf_ini  (comm,                          /**/ &e->cf));
    UC(eobj_unpackf_ini(L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->uf));
}

void obj_inter_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, float dt, int maxp, /**/ ObjInter **oi) {
    int rank;
    ObjInter *o;
    int3 L = opt->params.L;
    EMALLOC(1, oi);
    o = *oi;

    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, L, &o->e));

    o->cnt = NULL;
    o->fsi = NULL;

    int nobj = opt->nmbr + opt->nrig;
    UC(cnt_ini(maxp, rank, L, nobj, /**/ &o->cnt));
    UC(fsi_ini(rank, L, /**/ &o->fsi));
}


static void fin_obj_exch(/**/ ObjExch *e) {
    UC(eobj_pack_fin   (/**/ e->p));
    UC(eobj_comm_fin   (/**/ e->c));
    UC(eobj_unpack_fin (/**/ e->u));

    UC(eobj_packf_fin  (/**/ e->pf));
    UC(eobj_commf_fin  (/**/ e->cf));
    UC(eobj_unpackf_fin(/**/ e->uf));
    EFREE(e);
}

void obj_inter_fin(ObjInter *o) {
    UC(fin_obj_exch(o->e));
    if (o->cnt) UC(cnt_fin(o->cnt));
    if (o->fsi) UC(fsi_fin(o->fsi));
    EFREE(o);
}
static void fill_wrappers(PFarrays *obj, int *nwrappers, PaWrap *pw, FoWrap *fw) {
    long i, nw, n;
    PaArray p;
    FoArray f;

    (*nwrappers) = nw = pfarrays_size(obj);

    if (nw > MAX_OBJ_TYPES)
        ERR("Too many objects in wrappers : %d/%d", nw, MAX_OBJ_TYPES);

    for (i = 0; i < nw; ++i, ++pw, ++fw) {
        UC(pfarrays_get(i, obj, &n, &p, &f));
        pw->n = fw->n = n;
        pw->pp = (const Particle*) p.pp;
        fw->ff = (Force*) f.ff;
    }
}

static void bind_solvent(PFarray *flu, const int *starts, Fsi *fsi) {
    UC(fsi_bind_solvent(flu->p, (Force*) flu->f.ff, flu->n, starts, /**/ fsi));
}

static void forces_cnt(ObjInter *oi, int nw, const PairParams *prms[], PaWrap *pw, FoWrap *fw) {
    cnt_build_cells(  nw, prms, pw, /**/ oi->cnt);
    cnt_bulk(oi->cnt, nw, prms, pw, fw);
}

static void forces_fsi(ObjInter *oi, int nw, const PairParams *prms[], PaWrap *pw, FoWrap *fw) {
    fsi_bulk(oi->fsi, nw, prms, pw, fw);
}

static bool has_fsi_work(int nw, const PairParams **prms) {
    for (int i = 0; i < nw; ++i)
        if (prms[i]) return true;
    return false;
}

static bool has_work(const ObjInter *o, int nw, const PairParams **fsi_prms) {
    return o->cnt || has_fsi_work(nw, fsi_prms);
}

void obj_inter_forces(ObjInter *oi, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj) {
    PaWrap pw[MAX_OBJ_TYPES];
    FoWrap fw[MAX_OBJ_TYPES];
    int nw = 0;
    ObjExch *e = oi->e;
    int all_counts[26 * MAX_OBJ_TYPES] = {0};

    UC(fill_wrappers(obj, /**/ &nw, pw, fw));
    if (!nw) return;
    if (!has_work(oi, nw, fsi_prms)) return;

    /* Prepare and send the data */

    UC(eobj_build_map(nw, pw, /**/ e->p));
    UC(eobj_pack(nw, pw, /**/ e->p));
    UC(eobj_download(nw, /**/ e->p));

    UC(eobj_post_send(e->p, e->c));
    UC(eobj_post_recv(e->c, e->u));

    /* bulk interactions */

    if (oi->fsi) UC(bind_solvent(flu, flu_start, oi->fsi));

    if (oi->fsi) UC(forces_fsi(oi, nw, fsi_prms, pw, fw));
    if (oi->cnt) UC(forces_cnt(oi, nw, cnt_prms, pw, fw));

    /* recv data and halo interactions  */

    UC(eobj_wait_send(e->c));
    UC(eobj_wait_recv(e->c, e->u));

    eobj_get_all_counts(nw, e->u, /**/ all_counts);
    Pap26 hpp = eobj_upload_shift(e->u);
    Fop26 hff = eobj_reini_ff(e->u, e->pf);

    if (oi->fsi) UC(fsi_halo(oi->fsi, nw, fsi_prms,         hpp, hff, all_counts));
    if (oi->cnt) UC(cnt_halo(oi->cnt, nw, cnt_prms, pw, fw, hpp, hff, all_counts));

    /* send the forces back */

    UC(eobj_download_ff(e->pf));

    UC(eobj_post_send_ff(e->pf, e->cf));
    UC(eobj_post_recv_ff(e->cf, e->uf));

    UC(eobj_wait_send_ff(e->cf));
    UC(eobj_wait_recv_ff(e->cf, e->uf));

    UC(eobj_unpack_ff(e->uf, e->p, nw, /**/ fw));
}






#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
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

#ifndef UDX_SIM_MESH_IO_COMMON
#define UDX_SIM_MESH_IO_COMMON
struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

#ifndef UDX_SIM_MESH_WRITE_COMMON
#define UDX_SIM_MESH_WRITE_COMMON
enum {MESH_WRITE_EDGE, MESH_WRITE_CENTER};
#endif

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory, /**/ MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
#endif

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON
/* solvent options */
struct OptFlu {
    bool colors, ids, ss, push;
};

/* common parameters for membranes and rigid objects */
struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};

/* membrane options */
struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};

/* rigid options */
struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct OptDump {
    bool       parts,      mesh,      field,      strt, forces;
    float freq_parts, freq_mesh, freq_field, freq_strt, freq_diag;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
};

struct Opt {
    OptFlu flu;
    OptMbr mbr[MAX_MBR_TYPES];
    OptRig rig[MAX_RIG_TYPES];
    int nmbr, nrig;
    OptWall wall;
    OptParams params;
    OptDump dump;
    bool inflow, outflow, denoutflow, vcon;
    int recolor_freq;
    int sampler_npdump;
    int3 sampler_grid_ref;
    bool restart;
    bool tracers;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
long opt_estimate_maxp(const Opt*);
#endif

static int str2shifttype(const char *type) {
    if      (same_str(type, "edge"  )) return MESH_WRITE_EDGE;
    else if (same_str(type, "center")) return MESH_WRITE_CENTER;
    else
        ERR("Unrecognised rbc shift type <%s>", type);
    return -1;
}

static int get_shifttype(const Config *c, const char *desc) {
    const char *type;
    UC(conf_lookup_string(c, desc, &type));
    return str2shifttype(type);
 }

static int get_shifttype_ns(const Config *c, const char *ns, const char *d) {
    const char *type;
    UC(conf_lookup_string_ns(c, ns, d, &type));
    return str2shifttype(type);
}

static void lookup_bool(const Config *c, const char *desc, bool *res) {
    int b;
    UC(conf_lookup_bool(c, desc, &b));
    *res = b;
}

static void lookup_bool_ns(const Config *c, const char *ns, const char *desc, bool *res) {
    int b;
    UC(conf_lookup_bool_ns(c, ns, desc, &b));
    *res = b;
}

static void lookup_string(const Config *c, const char *desc, char *res) {
    const char *s;
    UC(conf_lookup_string(c, desc, &s));
    strcpy(res, s);
}

static void lookup_string_ns(const Config *c, const char *ns, const char *desc, char *res) {
    const char *s;
    UC(conf_lookup_string_ns(c, ns, desc, &s));
    strcpy(res, s);
}

static void read_flu(const Config *c, OptFlu *o) {
    UC(lookup_bool(c, "flu.colors", &o->colors));
    UC(lookup_bool(c, "flu.ids", &o->ids));
    UC(lookup_bool(c, "flu.stresses", &o->ss));
    UC(lookup_bool(c, "flu.push", &o->push));
}

static void read_obj_common(const Config *c, bool restart, const char *ns, OptObj *o) {
    o->shifttype = get_shifttype_ns(c, ns, "shifttype");
    UC(lookup_bool_ns(c, ns, "push",   &o->push));
    UC(lookup_bool_ns(c, ns, "bounce", &o->bounce));
    UC(conf_lookup_float_ns(c, ns, "mass", &o->mass));

    UC(lookup_string_ns(c, ns, "templ_file", o->templ_file));

    if (restart) o->ic_file[0] = '\0';
    else         UC(lookup_string_ns(c, ns, "ic_file", o->ic_file));

    strcpy(o->name, ns);
}

static void read_mbr(const Config *c, bool restart, const char *ns, OptMbr *o) {
    UC(read_obj_common(c, restart, ns, o));
    UC(lookup_bool_ns(c, ns, "ids", &o->ids));
    UC(lookup_bool_ns(c, ns, "stretch", &o->stretch));

    UC(lookup_bool(c, "dump.rbc_com", &o->dump_com));
    UC(conf_lookup_int_ns(c, ns, "substeps", &o->substeps));

    if (o->stretch)
        UC(lookup_string_ns(c, ns, "stretch_file", o->stretch_file));
}

static void read_mbr_array(const Config *c, bool restart, int *nmbr, OptMbr *oo) {
    int i, n;
    const char *ss[MAX_MBR_TYPES];
    UC(conf_lookup_vstring(c, "membranes", MAX_MBR_TYPES, &n, ss));
    for (i = 0; i < n; ++i)
        UC(read_mbr(c, restart, ss[i], &oo[i]));
    *nmbr = n;
}

static void read_rig(const Config *c, bool restart, const char *ns, OptRig *o) {
    UC(read_obj_common(c, restart, ns, o));
    UC(lookup_bool_ns(c, ns, "empty_pp", &o->empty_pp));
}

static void read_rig_array(const Config *c, bool restart, int *nrig, OptRig *oo) {
    int i, n;
    const char *ss[MAX_RIG_TYPES];
    UC(conf_lookup_vstring(c, "rigids", MAX_RIG_TYPES, &n, ss));
    for (i = 0; i < n; ++i)
        UC(read_rig(c, restart, ss[i], &oo[i]));
    *nrig = n;
}

static void read_wall(const Config *c, OptWall *o) {
    UC(lookup_bool(c, "wall.active", &o->active));
}

static void read_params(const Config *c, OptParams *p) {
    UC(conf_lookup_int3 (c, "glb.L",          &p->L         ));
    UC(conf_lookup_float(c, "glb.kBT",        &p->kBT       ));
    UC(conf_lookup_int  (c, "glb.numdensity", &p->numdensity));
}

static void read_dump(const Config *c, OptDump *o) {
    UC(lookup_string(c, "dump.strt_base_dump", o->strt_base_dump));
    UC(lookup_string(c, "dump.strt_base_read", o->strt_base_read));

    UC(lookup_bool(c, "dump.parts", &o->parts));
    UC(conf_lookup_float(c, "dump.freq_parts", &o->freq_parts));

    UC(lookup_bool(c, "dump.mesh", &o->mesh));
    UC(conf_lookup_float(c, "dump.freq_mesh", &o->freq_mesh));

    UC(lookup_bool(c, "dump.field", &o->field));
    UC(conf_lookup_float(c, "dump.freq_field", &o->freq_field));

    UC(lookup_bool(c, "dump.strt", &o->strt));
    UC(conf_lookup_float(c, "dump.freq_strt", &o->freq_strt));

    UC(lookup_bool(c, "dump.forces", &o->forces));
    UC(conf_lookup_float(c, "dump.freq_diag", &o->freq_diag));
}

static void read_common(const Config *c, Opt *o) {
    UC(conf_lookup_int(c, "sampler.n_per_dump", &o->sampler_npdump));
    UC(conf_lookup_int3(c, "sampler.grid_ref", &o->sampler_grid_ref));

    UC(lookup_bool(c, "outflow.active", &o->outflow));
    UC(lookup_bool(c, "inflow.active", &o->inflow));
    UC(lookup_bool(c, "denoutflow.active", &o->denoutflow));
    UC(lookup_bool(c, "vcon.active", &o->vcon));

    UC(conf_lookup_int(c, "flu.recolor_freq", &o->recolor_freq));

    UC(lookup_bool(c, "glb.restart", &o->restart));
}

void opt_read(const Config *c, Opt *o) {
    UC(read_common(c, o));
    UC(read_params(c, &o->params));
    UC(read_dump(c, &o->dump));
    UC(read_flu(c, &o->flu));

    UC(read_mbr_array(c, o->restart, &o->nmbr, o->mbr));
    UC(read_rig_array(c, o->restart, &o->nrig, o->rig));

    UC(read_wall(c, &o->wall));
}

static void check_mbr(const OptMbr *m) {
    enum {
        MIN_SUBSTEPS = 1,
        MAX_SUBSTEPS = 1000
    };
    if (m->dump_com && !m->ids)
        ERR("%s: Need ids activated to dump com!", m->name);

    if (m->substeps < MIN_SUBSTEPS || m->substeps > MAX_SUBSTEPS)
        ERR("%s: substeps out of range: %d (min: %d, max: %d)",
            m->name, m->substeps, MIN_SUBSTEPS, MAX_SUBSTEPS);
}

void opt_check(const Opt *o) {
    for (int i = 0; i < o->nmbr; ++i)
        check_mbr(&o->mbr[i]);
}

static long maxp_estimate(const OptParams *p) {
    int3 L = p->L;
    int flu, obj, estimate;
    flu = L.x * L.y * L.z * p->numdensity;
    obj = MAX_OBJ_TYPES * MAX_PSOLID_NUM * MAX_SOLIDS;
    estimate = flu > obj ? flu : obj;
    return SAFETY_FACTOR_MAXP * estimate;
}

long opt_estimate_maxp(const Opt *o) {
    return maxp_estimate(&o->params);
}






#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
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
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

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

#ifndef UDX_SIM_OBJECTS_PARRAY_COMMON
#define UDX_SIM_OBJECTS_PARRAY_COMMON
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
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
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
#endif
#ifndef UDX_SIM_OBJECTS_PFARRAYS_COMMON
#define UDX_SIM_OBJECTS_PFARRAYS_COMMON
struct PFarray {
    PaArray p;
    FoArray f;
    long n;
};

struct PFarrays;

struct PaArray;
struct FoArray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarrays_clear(PFarrays*);
void pfarrays_push(PFarrays*, long n, PaArray, FoArray);
void pfarrays_push(PFarrays*, PFarray);
int  pfarrays_size(const PFarrays*);
void pfarrays_get(int i, const PFarrays*, long *n, PaArray*, FoArray*);
void pfarrays_get(int i, const PFarrays*, PFarray*);
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
struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};

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
#ifndef UDX_SIM_WALLS_COMMON
#define UDX_SIM_WALLS_COMMON
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
#endif

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON
/* solvent options */
struct OptFlu {
    bool colors, ids, ss, push;
};

/* common parameters for membranes and rigid objects */
struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};

/* membrane options */
struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};

/* rigid options */
struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct OptDump {
    bool       parts,      mesh,      field,      strt, forces;
    float freq_parts, freq_mesh, freq_field, freq_strt, freq_diag;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
};

struct Opt {
    OptFlu flu;
    OptMbr mbr[MAX_MBR_TYPES];
    OptRig rig[MAX_RIG_TYPES];
    int nmbr, nrig;
    OptWall wall;
    OptParams params;
    OptDump dump;
    bool inflow, outflow, denoutflow, vcon;
    int recolor_freq;
    int sampler_npdump;
    int3 sampler_grid_ref;
    bool restart;
    bool tracers;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
long opt_estimate_maxp(const Opt*);
#endif

struct Wall;

struct OptParams;
struct Config;
struct PairParams;
struct Coords;

void wall_ini(const Config*, int3 L, Wall**);
void wall_fin(Wall*);

void wall_gen(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
              /*io*/ int *n, Particle *pp, /**/ Wall*);

void wall_restart(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
                  const char *base, /**/ Wall*);

void wall_dump_templ(const Wall*, MPI_Comm, const char *base);

void wall_interact(const Coords*, const PairParams*[], Wall*, PFarrays*);
void wall_adhesion(const Coords*, const PairParams*[], Wall*, PFarrays*);
void wall_bounce(const Wall*, const Coords*, float dt, PFarrays*);
void wall_repulse(const Wall*, const WallRepulsePrm *[], PFarrays*);

void wall_update_vel(float t, Wall *);

void wall_get_sdf_ptr(const Wall*, const Sdf**);           // <1>
double wall_compute_volume(const Wall*, MPI_Comm, int3 L); // <2>
struct Wall {
    Sdf *sdf;
    WallQuants q;
    WallTicket *t;
    Wvel *vel;
    WvelStep *velstep;
};
void wall_ini(const Config *cfg, int3 L, Wall **wall) {
    Wall *w;
    EMALLOC(1, wall);
    w = *wall;
    UC(sdf_ini(L, &w->sdf));
    UC(wall_ini_quants(L, &w->q));
    UC(wall_ini_ticket(L, &w->t));
    UC(wvel_ini(&w->vel));
    UC(wvel_set_conf(cfg, w->vel));
    UC(wvel_step_ini(&w->velstep));
}

void wall_fin(Wall *w) {
    UC(sdf_fin(w->sdf));
    UC(wall_fin_quants(&w->q));
    UC(wall_fin_ticket(w->t));
    UC(wvel_fin(w->vel));
    UC(wvel_step_fin(w->velstep));
    EFREE(w);
}
static long get_max_parts(OptParams p) {
    int3 L = p.L;
    return p.numdensity *
        (L.x + 2 * XWM) *
        (L.y + 2 * YWM) *
        (L.z + 2 * ZWM);
}

void wall_gen(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
              /*io*/ int *n, Particle *pp, /**/ Wall *w) {
    long maxn, nold, nnew;
    if (!w) return;

    maxn = get_max_parts(op);
    nold = *n;
    UC(sdf_gen(coords, cart, dump_sdf, /**/ w->sdf));
    MC(m::Barrier(cart));
    UC(wall_gen_quants(cart, op.numdensity, maxn, w->sdf, /**/ n, pp, &w->q));
    if (w->q.n) UC(wall_gen_ticket(&w->q, w->t));
    nnew = *n;
    msg_print("solvent particles survived: %d/%d", nnew, nold);
}

void wall_restart(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
                  const char *base, /**/ Wall *w) {
    long maxn;
    if (!w) return;

    maxn = get_max_parts(op);
    UC(sdf_gen(coords, cart, dump_sdf, /**/ w->sdf));
    MC(m::Barrier(cart));
    UC(wall_strt_quants(cart, base, maxn, &w->q));
    UC(wall_gen_ticket(&w->q, w->t));
}

void wall_dump_templ(const Wall *w, MPI_Comm cart, const char *base) {
    UC(wall_strt_dump_templ(cart, base, &w->q));
}

void wall_get_sdf_ptr(const Wall *w, const Sdf **s) {*s = w->sdf;}

double wall_compute_volume(const Wall *w, MPI_Comm comm, int3 L) {
    enum {NSAMPLES = 100000};
    return sdf_compute_volume(comm, L, w->sdf, NSAMPLES);
}

void wall_interact(const Coords *coords, const PairParams *prms[], Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const PairParams *prm;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        prm = prms[i];
        if (!n)   continue;
        if (!prm) continue;
        UC(wall_force(prm, w->velstep, coords, w->sdf, &w->q, w->t, n, &p, /**/ &f));
    }
}

void wall_adhesion(const Coords *coords, const PairParams* params[], Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const PairParams *par;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        par = params[i];

        if (!par) continue;
        if (!n)   continue;

        UC(wall_force_adhesion(par, w->velstep, coords, w->sdf, &w->q, w->t, n, &p, /**/ &f));
    }
}

void wall_bounce(const Wall *w, const Coords *coords, float dt, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        if (n) UC(sdf_bounce(dt, w->velstep, coords, w->sdf, n, /**/ (Particle*) p.pp));
    }
}

void wall_repulse(const Wall *w, const WallRepulsePrm *params[], PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const WallRepulsePrm *wrp;

    na = pfarrays_size(aa);
    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        wrp = params[i];

        if (!n)   continue;
        if (!wrp) continue;

        UC(wall_repulse(w->sdf, wrp, n, &p, /**/ &f));
    }
}

void wall_update_vel(float t, Wall *w) {
    UC(wvel_get_step(t, w->vel, /**/ w->velstep));
}






// Grouped editable compilation unit for I/O and half-edge code.


// Grouped editable compilation unit for output, mesh, and restart I/O.


// Grouped editable compilation unit for all current output backends.


// Grouped editable compilation unit for BOP, VTK, and point-grid output.


// Grouped editable compilation unit for BOP and VTK output.



#include <mpi.h>
#include <string.h>
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

/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
#endif /* UDX_SIM_SCHEME_DEVICE_API_COMMON */

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

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif /* UDX_SIM_OBJECTS_DEF_COMMON */
void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
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

struct IoBop;

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **t);
void io_bop_fin(IoBop *t);

void io_bop_parts       (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t);
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t);
void io_bop_stresses    (MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t);
void io_bop_ids         (MPI_Comm cart, long n, const int   *ii, const char *name, int id, IoBop *t);
void io_bop_colors      (MPI_Comm cart, long n, const int   *cc, const char *name, int id, IoBop *t);
struct IoBop {
    BopData *ff; /* float data */
    BopData *ii; /*   int data */
};
enum {
    NVARP = 6,
    NVARF = 3,
    NVARS = 6,
    NVARI = 1,
    NVARC = 1
};

#define VARP "x y z vx vy vz"
#define VARF "fx fy fz"
#define VARS "sxx sxy sxz syy syz szz"
#define VARI "ids"
#define VARC "color"

#define PATTERN "%s-%05d"

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **ib) {
    IoBop *t;
    EMALLOC(1, ib);
    t = *ib;

    BPC( bop_ini(&t->ff) );
    BPC( bop_ini(&t->ii) );

    BPC( bop_set_n(maxp, t->ff) );
    BPC( bop_set_n(maxp, t->ii) );

    BPC( bop_set_vars(NVARP + NVARF, VARP " " VARF, t->ff) );
    BPC( bop_set_vars(NVARI, VARI, t->ii) );

    BPC( bop_set_type(BopFLOAT, t->ff) );
    BPC( bop_set_type(  BopINT, t->ii) );

    BPC( bop_alloc(t->ff) );
    BPC( bop_alloc(t->ii) );

    if (m::is_master(comm))
        UC(os_mkdir(DUMP_BASE "/bop"));
}

void io_bop_fin(IoBop *t) {
    BPC( bop_fin(t->ff) );
    BPC( bop_fin(t->ii) );
    EFREE(t);
}

static void p2f3(const Particle p, /**/ float3 *r, float3 *v) {
    enum {X, Y, Z};
    r->x = p.r[X];
    r->y = p.r[Y];
    r->z = p.r[Z];

    v->x = p.v[X];
    v->y = p.v[Y];
    v->z = p.v[Z];
}

static void f2f3(const Force fo, /**/ float3 *f) {
    enum {X, Y, Z};
    f->x = fo.f[X];
    f->y = fo.f[Y];
    f->z = fo.f[Z];
}

static void copy_shift(const Coords *c, long n, const Particle *pp, /**/ float3 *w) {
    float3 r, v;
    for (int j = 0; j < n; ++j) {
        p2f3(pp[j], /**/ &r, &v);
        local2global(c, r, /**/ &w[2*j]);
        w[2*j + 1] = v;
    }
}

static void copy_shift_with_forces(const Coords *c, long n, const Particle *pp, const Force *ff, /**/ float3 *w) {
    float3 r, v, f;
    for (int j = 0; j < n; ++j) {
        p2f3(pp[j], /**/ &r, &v);
        f2f3(ff[j], /**/ &f);
        local2global(c, r, /**/ &w[3*j]);
        w[3*j + 1] = v;
        w[3*j + 2] = f;
    }
}

static void set_name(const char *base, int id, /**/ char *name) {
    sprintf(name, DUMP_BASE "/bop/" PATTERN, base, id);
}

static void write(MPI_Comm cart, const char *base, int id, BopData *bop) {
    char fname[FILENAME_MAX];
    set_name(base, id, fname);

    BPC( bop_write_header(cart, fname, bop) );
    BPC( bop_write_values(cart, fname, bop) );
}

void io_bop_parts (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t) {
    BopData *bop  = t->ff;
    float3 *ppout = (float3*) bop_get_data(bop);

    copy_shift(coords, n, pp, /**/ ppout);
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARP, VARP, bop) );

    write(cart, name, id, bop);
}

void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t) {
    BopData *bop = t->ff;
    float3 *out  = (float3*) bop_get_data(bop);

    copy_shift_with_forces(coords, n, pp, ff, /**/ out);
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARP + NVARF, VARP " " VARF, bop) );

    write(cart, name, id, bop);
}

void io_bop_stresses(MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t) {
    BopData *bop = t->ff;
    float   *out = (float*) bop_get_data(bop);

    memcpy(out, ss, n * NVARS * sizeof(float));
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARS, VARS, bop) );

    write(cart, name, id, bop);
}

void io_bop_ids(MPI_Comm cart, long n, const int *ii, const char *name, int id, IoBop *t) {
    BopData *bop = t->ii;
    int     *out = (int*) bop_get_data(bop);

    memcpy(out, ii, n * NVARI * sizeof(int));
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARI, VARI, bop) );

    write(cart, name, id, bop);
}

void io_bop_colors(MPI_Comm cart, long n, const int *cc, const char *name, int id, IoBop *t) {
    BopData *bop = t->ii;
    int     *out = (int*) bop_get_data(bop);

    memcpy(out, cc, n * NVARC * sizeof(int));
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARC, VARC, bop) );

    write(cart, name, id, bop);
}

#undef VARP
#undef VARF
#undef VARS
#undef VARI
#undef VARC

#undef PATTERN




// Grouped editable compilation unit for VTK output.



#include <stdio.h>
#include <vector_types.h>

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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);

struct Mesh;
struct MeshRead;

void mesh_ini(MeshRead*, /**/ Mesh**);
void mesh_copy(const Mesh*, /**/ Mesh**);
void mesh_fin(Mesh*);

int mesh_nv(Mesh*);
int mesh_nt(Mesh*);
int mesh_ne(Mesh*);
const int* mesh_tt(Mesh*);

struct Mesh {
    int nv, nt, ne; /* vertices and triangles */
    int *tt;   /* triangles [v0 v1 v2] [v0 v1 v2] .. */
    int *ee;   /* edges [e0 e1] [e0 e1] .. */
};
void mesh_ini(MeshRead *mesh, /**/ Mesh **pq) {
    Mesh *q;
    int i, j, nv, nt, ne;
    const int4 *tt, *dd;
    int4 t, d;
    EMALLOC(1, &q);

    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    ne = mesh_read_get_ne(mesh);
    tt = mesh_read_get_tri(mesh);
    dd = mesh_read_get_dih(mesh);

    EMALLOC(3*nt, &q->tt);
    EMALLOC(2*ne, &q->ee);
    for (i = j = 0; i < nt; i++) {
        t = tt[i];
        q->tt[j++] = t.x; q->tt[j++] = t.y; q->tt[j++] = t.z;
    }

    for (i = j = 0; i < ne; i++) {
        d = dd[i];
        q->ee[j++] = d.y; q->ee[j++] = d.z;
    }

    q->nv = nv; q->nt = nt; q->ne = ne;
    *pq = q;
}

void mesh_copy(const Mesh *f, /**/ Mesh **pt) {
    Mesh *t; /* to, from */
    int nv, nt, ne;
    EMALLOC(1, &t);

    nv = f->nv; nt = f->nt; ne = f->ne;

    EMALLOC(3*nt, &t->tt);
    EMALLOC(2*ne, &t->ee);

    EMEMCPY(3*nt, f->tt, t->tt);
    EMEMCPY(2*ne, f->ee, t->ee);

    t->nv = nv; t->nt = nt; t->ne = ne;
    *pt = t;
}

void mesh_fin(Mesh *q) {
    EFREE(q->tt);
    EFREE(q->ee);
    EFREE(q);
}

int mesh_nv(Mesh *q) { return q->nv; }
int mesh_nt(Mesh *q) { return q->nt; }
int mesh_ne(Mesh *q) { return q->ne; }
const int* mesh_tt(Mesh *q) { return q->tt; }





#include <stdio.h>
#include <stdarg.h>
#include <string.h>
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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO

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

struct KeyList;

void key_list_ini(KeyList**);
void key_list_copy(KeyList*, /**/ KeyList**);
void key_list_fin(KeyList*);

void key_list_push(KeyList*, const char*);
int key_list_has(KeyList*, const char*);
int key_list_offset(KeyList*, const char*);
int key_list_width(KeyList*, const char*);
int key_list_size(KeyList*);
const char* key_list_name(KeyList*, int i);

void key_list_mark(KeyList*, const char*);
int  key_list_marked(KeyList*);
void key_list_unmark(KeyList*);

void key_list_log(KeyList*);
struct Particle;
struct Vectors;
struct Coords;

void vectors_float_ini(int n, const float*, /**/ Vectors**); // <1>
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**); // <2>
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*, /**/ Vectors**); // <3>
void vectors_postions_center_ini(const Coords*, int n, const Particle*, /**/ Vectors**); // <4>
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**); // <5>
void vectors_zero_ini(int n, /**/ Vectors**); // <6>

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i, /**/ float r[3]); // <7>
struct Particle;
struct Scalars;
struct Vectors;

void scalars_float_ini(int n, const float*, /**/ Scalars**);
void scalars_double_ini(int n, const double*, /**/ Scalars**);
void scalars_vectors_ini(int n, const Vectors*, int dim, /**/ Scalars**);

void scalars_zero_ini(int n, /**/ Scalars**);
void scalars_one_ini(int n, /**/ Scalars**);

void scalars_fin(Scalars*);
double scalars_get(const Scalars*, int i);

struct Mesh;
struct MeshRead;

void mesh_ini(MeshRead*, /**/ Mesh**);
void mesh_copy(const Mesh*, /**/ Mesh**);
void mesh_fin(Mesh*);

int mesh_nv(Mesh*);
int mesh_nt(Mesh*);
int mesh_ne(Mesh*);
const int* mesh_tt(Mesh*);
struct WriteFile;

int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*); // <1>
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*); // <2>

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);

struct VTK;
struct VTKConf;

struct MeshRead;
struct Vectors;
struct Scalars;

void vtk_conf_ini(MeshRead*, /**/ VTKConf**); // <1>
void vtk_conf_fin(VTKConf*);
void vtk_conf_vert(VTKConf*, const char *keys); // <2>
void vtk_conf_tri(VTKConf*, const char *keys); // <3>

void vtk_ini(MPI_Comm, int maxm, char const *dir, VTKConf*, /**/ VTK**); // <1>
void vtk_points(VTK*, int nm, const Vectors*); // <2>

void vtk_vert(VTK*, int nm, const Scalars*, const char *keys); // <3>
void vtk_tri(VTK*, int nm, const Scalars*, const char *keys);  // <4>

void vtk_write(VTK*, MPI_Comm, int id); // <5>
void vtk_fin(VTK*);

struct VTKConf {
    Mesh *mesh;
    KeyList *tri, *vert;
};

enum { MAGIC = 1234 };
#ifndef UDX_IO_OUTPUT_COMMON_LIMITS
#define UDX_IO_OUTPUT_COMMON_LIMITS
#ifndef UDX_IO_OUTPUT_COMMON_UNSET
#define UDX_IO_OUTPUT_COMMON_UNSET
enum { UNSET = - 1 };
#endif
enum { N_MAX = 100 };
#endif
struct VTK {
    int nm; /* current number of meshes */
    Mesh *mesh;
    KeyList *tri, *vert;

    int nbuf; /* maximum bufer size */
    double *dbuf;
    int *ibuf;

    double *TRI[N_MAX];
    double *VERT[N_MAX];

    int rr_set;
    char path[FILENAME_MAX];

    int stamp; /* MAGIC */
};

struct Out {
    WriteFile *file;
    MPI_Comm comm;
};
#ifndef UDX_IO_OUTPUT_COMMON_UTIL
#define UDX_IO_OUTPUT_COMMON_UTIL
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
#endif

#ifndef UDX_IO_OUTPUT_MKDIR_COMMON
#define UDX_IO_OUTPUT_MKDIR_COMMON
static void mkdir(MPI_Comm comm, const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    if (!m::is_master(comm)) return;
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}
#endif

static int max3(int a, int b, int c) {
    if (a > b) b = a;
    if (b > c) c = b;
    return c;
}
static int get_nbuf(int nm, int nv, int nt, int ne) {
    return max3(3*nm*nv, 4*nm*nt, 2*nm*ne);
}

static int little_p0() {
    int a;
    unsigned char *b;
    a = 1;
    b = (unsigned char*)&a;
    return (*b) != 0;
}

static int little_p() {
    static int done = 0;
    static int little = 0;
    if (done) return little;
    little = little_p0();
    done = 1;
    return little;
}

static void big_endian_dbl0(double *px) {
    const int n = sizeof(double);
    int i;
    unsigned char *c, b[n];
    if (!little_p()) return;
    c = (unsigned char*)px;
    for (i = 0; i < n; i++) b[i] = c[i];
    for (i = 0; i < n; i++) c[i] = b[n - i - 1];
}

static void big_endian_dbl(int n, double *d) {
    int i;
    for (i = 0; i < n; i++)
        big_endian_dbl0(&d[i]);
}

static void big_endian_int0(int *px) {
    const int n = sizeof(int);
    int i;
    unsigned char *c, b[n];
    if (!little_p()) return;
    c = (unsigned char*)px;
    for (i = 0; i < n; i++) b[i] = c[i];
    for (i = 0; i < n; i++) c[i] = b[n - i - 1];
}

static void big_endian_int(int n, int *d) {
    int i;
    for (i = 0; i < n; i++)
        big_endian_int0(&d[i]);
}
enum { BUF = 9999 };

static void print(Out *out, const char *fmt, ...) {
    char s[BUF];
    int size;
    va_list args;
    size = 0;
    if (m::is_master(out->comm)) {
        va_start(args, fmt);
        size = vsnprintf(s, BUF - 1, fmt, args);
        va_end(args);
    }
    UC(write_master(out->comm, s, size, out->file));
}
static void header(Out *o) {
    print(o, "# vtk DataFile Version 2.0\n");
    print(o, "created with uDeviceX\n");
    print(o, "BINARY\n");
    print(o, "DATASET POLYDATA\n");
}

static void points(Out *o, int n, double *rr) {
    int n_total;
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    big_endian_dbl(3*n, /**/ rr);
    UC(write_reduce(comm, n, &n_total));
    print(o, "POINTS %d double\n", n_total);
    UC(write_all(comm, rr, 3*n*sizeof(rr[0]), file));
    print(o, "\n");
}

static void tri(Out *o, int nm, int nv, int nt, const int *tt, int *buf) {
    enum {NVP = 3};
    int n, m, t, i, shift, nm_total;
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    n = nm * nv;
    UC(write_shift_indices(comm, n, &shift));
    for (t = m = 0; m < nm; m++)
        for (i = 0; i < nt; i++) {
            buf[t++] = NVP;
            buf[t++] = shift + nv*m + tt[3*i + 0];
            buf[t++] = shift + nv*m + tt[3*i + 1];
            buf[t++] = shift + nv*m + tt[3*i + 2];
        }
    big_endian_int(t, /**/ buf);
    UC(write_reduce(comm, nm, &nm_total));
    print(o, "POLYGONS %d %d\n", nt*nm_total, 4*nt*nm_total);
    UC(write_all(comm, buf, t*sizeof(buf[0]), file));
    print(o, "\n");
}

static void cell_header(Out *o, int n) {
    int n_total;
    MPI_Comm comm;
    comm = o->comm;
    UC(write_reduce(comm, n, &n_total));
    print(o, "CELL_DATA %d\n", n_total);
}

static void cell_data(Out *o, int n, double *data, const char *name) {
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    print(o, "SCALARS %s double 1\n", name);
    print(o, "LOOKUP_TABLE default\n");

    big_endian_dbl(n, /**/ data);
    UC(write_all(comm, data, n*sizeof(data[0]), file));
    print(o, "\n");
}

static void point_header(Out *o, int n) {
    int n_total;
    MPI_Comm comm;
    comm = o->comm;
    UC(write_reduce(comm, n, &n_total));
    print(o, "POINT_DATA %d\n", n_total);
}

static void point_data(Out *o, int n, double *data, const char *name) {
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    print(o, "SCALARS %s double 1\n", name);
    print(o, "LOOKUP_TABLE default\n");

    big_endian_dbl(n, /**/ data);
    UC(write_all(comm, data, n*sizeof(data[0]), file));
    print(o, "\n");
}
#define PATTERN "%s/%s/%05d.vtk"

void vtk_conf_ini(MeshRead *mesh, /**/ VTKConf **pq) {
    VTKConf *q;
    EMALLOC(1, &q);
    mesh_ini(mesh, &q->mesh);
    key_list_ini(&q->tri);
    key_list_ini(&q->vert);
    *pq = q;
}

void vtk_conf_fin(VTKConf *q) {
    key_list_fin(q->tri);
    key_list_fin(q->vert);
    mesh_fin(q->mesh);
    EFREE(q);
}

void vtk_conf_tri(VTKConf *q, const char *keys) {
    int n;
    n = key_list_size(q->tri);
    if (n > N_MAX)
        ERR("n=%d > N_MAX=%d", n, N_MAX);
    UC(key_list_push(q->tri, keys));
}

void vtk_conf_vert(VTKConf *q, const char *keys) {
    int n;
    n = key_list_size(q->tri);
    if (n > N_MAX)
        ERR("n=%d > N_MAX=%d", n, N_MAX);
    UC(key_list_push(q->vert, keys));
}

void vtk_ini(MPI_Comm comm, int maxm, char const *path, VTKConf *c, /**/ VTK **pq) {
    int i;
    VTK *q;
    int nbuf, nk;
    Mesh *mesh;
    EMALLOC(1, &q);
    mesh = c->mesh;
    nbuf = get_nbuf(maxm, mesh_nv(mesh), mesh_nt(mesh), mesh_ne(mesh));
    EMALLOC(nbuf, &q->dbuf);
    EMALLOC(nbuf, &q->ibuf);
    UC(mkdir(comm, DUMP_BASE, path));
    cpy(q->path, path);
    UC(nk = key_list_size(c->tri));
    for (i = 0; i < nk; i++) EMALLOC(nbuf, &q->TRI[i]);
    UC(nk = key_list_size(c->vert));
    for (i = 0; i < nk; i++) EMALLOC(nbuf, &q->VERT[i]);

    q->stamp = MAGIC;
    q->nbuf = nbuf;
    q->nm       = UNSET;
    q->rr_set   = 0;
    UC(mesh_copy(mesh, /**/ &q->mesh));
    UC(key_list_copy(c->tri, &q->tri));
    UC(key_list_copy(c->vert, &q->vert));

    *pq = q;
}

void vtk_points(VTK *q, int nm, const Vectors *pos) {
    enum {X, Y, Z};
    float r[3];
    int nv, n, i, j;
    if (q->nm != UNSET && q->nm != nm)
        ERR("q->nm=%d  != nm=%d", q->nm, nm);

    nv = mesh_nv(q->mesh);
    n = nv * nm;
    for (i = j = 0; i < n; i++) {
        UC(vectors_get(pos, i, r));
        q->dbuf[j++] = r[X];
        q->dbuf[j++] = r[Y];
        q->dbuf[j++] = r[Z];
    }

    q->nm = nm;
    q->rr_set = 1;
}

void vtk_tri(VTK *q, int nm, const Scalars *sc, const char *keys) {
    int i, j, nt, n;
    if (!key_list_has(q->tri, keys)) {
        msg_print("unkown key '%s'", keys);
        key_list_log(q->tri);
        ERR("");
    }
    j = key_list_offset(q->tri, keys);
    nt = mesh_nt(q->mesh);
    n = nm * nt;
    for (i = 0; i < n; i++)
        q->TRI[j][i] = scalars_get(sc, i);
    UC(key_list_mark(q->tri, keys));
}

void vtk_vert(VTK *q, int nm, const Scalars *sc, const char *keys) {
    int i, j, nv, n;
    if (!key_list_has(q->vert, keys)) {
        msg_print("unkown key '%s'", keys);
        key_list_log(q->vert);
        ERR("wrong vtk_vert call");
    }
    j = key_list_offset(q->vert, keys);
    nv = mesh_nv(q->mesh);
    n = nm * nv;
    for (i = 0; i < n; i++)
        UC(q->VERT[j][i] = scalars_get(sc, i));
    UC(key_list_mark(q->vert, keys));
}

void vtk_write(VTK *q, MPI_Comm comm, int id) {
    char path[FILENAME_MAX] = {'\0'};
    int i, nk, n, nm, nv, nt, nbuf;
    const int *tt;
    const char *keys;
    Out out;
    if (q->stamp != MAGIC)
        ERR("VTK is not initialized");

    if (!key_list_marked(q->tri)) {
        msg_print("missing triangle data");
        key_list_log(q->tri);
        ERR("vtk_write is called is not enough data");
    }
    if (!key_list_marked(q->vert)) {
        msg_print("missing vertices data");
        key_list_log(q->vert);
        ERR("vtk_write is called is not enough data");
    }
    if (!q->rr_set) ERR("points are unset");

    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, q->path, id) < 0)
        ERR("snprintf failed");
    out.comm = comm;
    UC(write_file_open(comm, path, &out.file));

    nm = q->nm; nbuf = q->nbuf; nv = mesh_nv(q->mesh);
    nt = mesh_nt(q->mesh); tt = mesh_tt(q->mesh);

    n = nm * nv;
    if (3 * n > nbuf)
        ERR("3*n=%d > nbuf=%d", n, nbuf);
    if (4 * nm * nt > nbuf)
        ERR("nm=%d * nt=%d > nbuf=%d", nm, nt, nbuf);

    header(&out);
    points(&out, n, q->dbuf);
    tri(&out, nm, nv, nt, tt, q->ibuf);

    UC(nk = key_list_size(q->tri));
    if (nk > 0) cell_header(&out, nm*nt);
    for (i = 0; i < nk; i++) {
        keys = key_list_name(q->tri, i);
        cell_data(&out, nm*nt, q->TRI[i], keys);
    }

    UC(nk = key_list_size(q->vert));
    if (nk > 0) point_header(&out, n);
    for (i = 0; i < nk; i++) {
        keys = key_list_name(q->vert, i);
        point_data(&out, n, q->VERT[i], keys);
    }

    UC(write_file_close(out.file));

    q->rr_set = 0;
    q->nm = UNSET;
    key_list_unmark(q->tri);
    key_list_unmark(q->vert);
}

void vtk_fin(VTK *q) {
    int i;
    if (q->stamp != MAGIC)
        ERR("VTK is not initialized");
    EFREE(q->dbuf);
    EFREE(q->ibuf);
    for (i = 0; i < key_list_size(q->tri); i++)
        EFREE(q->TRI[i]);
    for (i = 0; i < key_list_size(q->vert); i++)
        EFREE(q->VERT[i]);
    key_list_fin(q->tri);
    key_list_fin(q->vert);
    EFREE(q);
}




#undef PATTERN


// Grouped editable compilation unit for point and grid output.


// Grouped editable compilation unit for point I/O and generic writing.



#include <string.h>
#include <ctype.h>
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


void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
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
int string_nword(const char *s);

/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
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

struct BopData;
#ifndef UDX_IO_OUTPUT_COMMON_LIMITS
#define UDX_IO_OUTPUT_COMMON_LIMITS
enum { N_MAX = 100 };
#endif

struct IOPointConf {
    int nn[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    int i;
};

#ifndef UDX_IO_OUTPUT_COMMON_UNSET
#define UDX_IO_OUTPUT_COMMON_UNSET
enum { UNSET = - 1 };
#endif
struct IOPoint {
    int maxn;
    int n;

    int nkey;
    int cum_n;
    int nn[N_MAX];
    int seen[N_MAX];
    char keys[N_MAX][FILENAME_MAX];

    char path[FILENAME_MAX];
    BopData *bop;
};
#ifndef UDX_IO_OUTPUT_COMMON_UTIL
#define UDX_IO_OUTPUT_COMMON_UTIL
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
#endif

static void mkdir(const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

static void reset(IOPoint *q) {
    int i, nkey;
    nkey = q->nkey;
    for (i = 0; i < nkey; i++)
        q->seen[i] = 0;
    q->n = UNSET;
}

static void wrong_key(IOPoint *q, const char *key) {
    int i, nkey;
    nkey = q->nkey;
    msg_print("unkown key: '%s'", key);
    msg_print("possible values:");
    for (i = 0; i < nkey; i++)
        msg_print("'%s'", q->keys[i]);
    ERR("");
}
#define PATTERN "%s/%s/%05d"

void io_point_conf_ini(/**/ IOPointConf **pq) {
    IOPointConf *q;
    EMALLOC(1, &q);
    q->i = 0;
    *pq = q;
}

static void conf_push(IOPointConf *q, int nv, const char *k0) {
    int i;
    i = q->i;
    cpy(q->keys[i], k0);
    q->nn[i] = nv; q->i = i + 1;
}
void io_point_conf_push(IOPointConf *q, const char *key) {
    int nw;
    UC(nw = string_nword(key));
    msg_print("string, nword: '%s', %d", key, nw);
    conf_push(q, nw, key);
}
void io_point_conf_fin(IOPointConf *q) { EFREE(q); }

static void ini_bop(int maxn, int n, const char *keys, BopData **pq) {
    BopData *q;
    BPC(bop_ini(&q));
    BPC(bop_set_n(maxn, q));
    BPC(bop_set_vars(n, keys, q));
    BPC(bop_set_type(BopDOUBLE, q));
    BPC(bop_alloc(q));

    *pq = q;
}
void io_point_ini(int maxn, const char *path, IOPointConf *c, /**/ IOPoint **pq) {
    int i, nkey, cum_n;
    IOPoint *q;
    char cum_key[N_MAX*(FILENAME_MAX + 1)];
    EMALLOC(1, &q);
    nkey = c->i;
    reset(q);

    cum_n = 0;
    for (i = 0; i < nkey; i++) {
        q->nn[i] = c->nn[i];
        cum_n += q->nn[i];
        cpy(q->keys[i], c->keys[i]);
    }

    for (i = 0; i < nkey; i++) {
        if (i > 0) cat(cum_key, " ");
        cat(cum_key, c->keys[i]);
    }

    ini_bop(maxn, cum_n, cum_key, &q->bop);

    UC(mkdir(DUMP_BASE, path));
    cpy(q->path, path);
    q->nkey = nkey;
    q->maxn = maxn;
    q->cum_n = cum_n;
    *pq = q;
}

void io_point_fin(IOPoint *q) {
    BPC(bop_fin(q->bop));
    EFREE(q);
}

static double *get_data(BopData *b) {
    BopType type;
    BPC(bop_get_type(b, &type));
    if (type != BopDOUBLE) ERR("BopType is not double");
    return (double*)bop_get_data(b);
}

static void push(int n, int nvar, int cum_n, const double *F, int offset, double *T) {
    int i, j, f, t; /* from/to */
    for (f = i = 0; i < n; i++) {
        for (j = 0; j < nvar; j++) {
            t = cum_n*i + offset + j;
            T[t] = F[f++];
        }
    }
}
void io_point_push(IOPoint *q, int n, const double *D, const char *key) {
    int offset, nkey, i;
    double *B;
    if (n > q->maxn)
        ERR("n=%d > q->maxn=%d; key: '%s'",
            n, q->maxn, key);

    if (q->n == UNSET)
        q->n = n;
    else if (q->n != n)
        ERR("q->n=%d != n=%d", q->n, n);

    nkey = q->nkey;
    offset = 0;
    for (i = 0; ; i++) {
        if (i == nkey) {UC(wrong_key(q, key)); ERR(""); }
        if (same_str(key, q->keys[i])) break;
        offset += q->nn[i];
    }
    if (q->seen[i]) ERR("seen key '%s' ", key);
    q->seen[i] = 1;
    UC(B = get_data(q->bop));
    push(n, q->nn[i], q->cum_n, D, offset, /**/ B);
}

void io_point_write(IOPoint *q, MPI_Comm comm, int id) {
    char name[FILENAME_MAX];
    int i, nkey, n;
    nkey = q->nkey;
    n = q->n;
    for (i = 0; i < nkey; i++)
        if (!q->seen[i])
            ERR("key '%s' was not pushed", q->keys[i]);

    if (snprintf(name, FILENAME_MAX, PATTERN, DUMP_BASE, q->path, id) < 0)
        ERR("snprintf failed");

    BPC(bop_set_n(n, q->bop));
    BPC(bop_write_header(comm, name, q->bop));
    BPC(bop_write_values(comm, name, q->bop));
    BPC(bop_summary(q->bop));

    reset(q);
}





#include <stdlib.h>
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
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

struct WriteFile;

int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*); // <1>
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*); // <2>

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);
struct WriteFile {
    MPI_File f;
};
int write_file_open(MPI_Comm comm, const char *fn, /**/ WriteFile **pq) {
    WriteFile *q;
    EMALLOC(1, &q);
    MC(m::File_open(comm, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &q->f));
    MC(m::File_set_size(q->f, 0));
    *pq = q;
    return 0;
}

int write_file_close(WriteFile *fp) {
    MPI_File f;
    f = fp->f;
    MC(m::File_close(&f));
    EFREE(fp);
    return 0;
}

void write_all(MPI_Comm comm, const void * const ptr, int nbytes32, WriteFile *fp) {
    MPI_File f = fp->f;
    MPI_Offset base;
    MPI_Offset offset = 0, nbytes = nbytes32;
    MPI_Status status;
    MPI_Offset ntotal = 0;
    MC(m::File_get_position(f, &base));
    MC(m::Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, comm));
    MC(m::File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MC(m::Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, comm) );
    MC(m::File_seek(f, ntotal, MPI_SEEK_CUR));
}

int write_master(MPI_Comm comm, const void * const ptr, int sz0, WriteFile *fp) {
    int sz;
    sz = m::is_master(comm) ? sz0 : 0;
    write_all(comm, ptr, sz, fp);
    return 0;
}

int write_shift_indices(MPI_Comm comm, int n, int *shift0) {
    *shift0 = 0;
    MC(m::Exscan(&n, shift0, 1, MPI_INTEGER, MPI_SUM, comm));
    return 0;
}

int write_reduce(MPI_Comm comm, int n0, /**/ int *n) {
    *n = 0;
    MC(m::Reduce(&n0, n, 1, MPI_INT, MPI_SUM, 0, comm));
    return 0;
}





// Grouped editable compilation unit for raw-grid and XMF output.



#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>

struct Coords;
void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c);
void coords_fin(Coords *c);
int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);
int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                           \
    } while (0)

#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();
void error_signal(const char *file, int line, const char *fmt, ...);
bool error_get();
void error_report();
void error_abort();

static int little_endian() {
    const uint16_t value = 1;
    return *((const uint8_t *) &value) == 1;
}

static uint32_t swap32(uint32_t value) {
    return ((value & 0x000000ffu) << 24) |
           ((value & 0x0000ff00u) << 8)  |
           ((value & 0x00ff0000u) >> 8)  |
           ((value & 0xff000000u) >> 24);
}

static void raw_path(const char *path, const char *name, char *out, size_t n) {
    size_t len = strlen(path);
    const char *suffix = ".raw";
    size_t suffix_len = strlen(suffix);
    int written;

    if (len >= suffix_len && strcmp(path + len - suffix_len, suffix) == 0)
        written = snprintf(out, n, "%.*s.%s.raw", (int) (len - suffix_len), path, name);
    else
        written = snprintf(out, n, "%s.%s.raw", path, name);
    if (written < 0 || (size_t) written >= n)
        ERR("raw grid path is too long");
}

static void raw_write_channel(const Coords *coords, MPI_Comm cart,
                              const char *path, const float *data) {
    int sizes[3] = { zdomain(coords), ydomain(coords), xdomain(coords) };
    int subsizes[3] = { zs(coords), ys(coords), xs(coords) };
    int starts[3] = { zlo(coords), ylo(coords), xlo(coords) };
    int local_count = subsizes[0] * subsizes[1] * subsizes[2];
    MPI_Datatype filetype;
    MPI_File file;
    MPI_Status status;
    float *swapped = NULL;
    const float *output = data;
    int i;
    int rc;
    char message[BUFSIZ];

    if (!little_endian()) {
        swapped = (float *) malloc((size_t) local_count * sizeof(*swapped));
        if (swapped == NULL) ERR("failed to allocate endian conversion buffer");
        for (i = 0; i < local_count; ++i) {
            uint32_t bits;
            memcpy(&bits, &data[i], sizeof(bits));
            bits = swap32(bits);
            memcpy(&swapped[i], &bits, sizeof(bits));
        }
        output = swapped;
    }

    rc = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                  MPI_FLOAT, &filetype);
    if (rc != MPI_SUCCESS) ERR("failed to create raw grid file view");
    rc = MPI_Type_commit(&filetype);
    if (rc != MPI_SUCCESS) ERR("failed to commit raw grid file view");

    rc = MPI_File_open(cart, path, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                       MPI_INFO_NULL, &file);
    if (rc != MPI_SUCCESS) ERR("failed to open raw grid file <%s>", path);
    rc = MPI_File_set_size(file, 0);
    if (rc != MPI_SUCCESS) ERR("failed to truncate raw grid file <%s>", path);
    rc = MPI_File_set_view(file, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    if (rc != MPI_SUCCESS) ERR("failed to set raw grid file view <%s>", path);
    rc = MPI_File_write_all(file, output, local_count, MPI_FLOAT, &status);
    if (rc != MPI_SUCCESS) {
        MPI_Error_string(rc, message, &i);
        message[i] = '\0';
        ERR("failed to write raw grid file <%s>: %s", path, message);
    }
    rc = MPI_File_close(&file);
    if (rc != MPI_SUCCESS) ERR("failed to close raw grid file <%s>", path);
    rc = MPI_Type_free(&filetype);
    if (rc != MPI_SUCCESS) ERR("failed to free raw grid file view");
    free(swapped);
}

void raw_write(int3 N, MPI_Comm cart, const char *path, int ncomp,
               const float **data, const char **names) {
    char channel_path[FILENAME_MAX];
    Coords *coords;
    int i;

    UC(coords_ini(cart, N.x, N.y, N.z, &coords));
    for (i = 0; i < ncomp; ++i) {
        raw_path(path, names[i], channel_path, sizeof(channel_path));
        raw_write_channel(coords, cart, channel_path, data[i]);
    }
    UC(coords_fin(coords));
}




#include <string.h>
#include <stdio.h>
#include <vector_types.h>

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

void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names);

static void header(FILE *f) {
    fprintf(f, "<?xml version=\"1.0\" ?>\n");
    fprintf(f, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(f, "<Xdmf Version=\"2.0\">\n");
    fprintf(f, " <Domain>\n");
}

static void epilogue(FILE *f) {
    fprintf(f, " </Domain>\n");
    fprintf(f, "</Xdmf>\n");
}

static float3 get_spacing(int3 D, int3 G) {
    float3 d;
    d.x = (float) D.x / (float) G.x;
    d.y = (float) D.y / (float) G.y;
    d.z = (float) D.z / (float) G.z;
    return d;
}

static void grid(int3 D, int3 G, FILE *f, const char *path, int n, const char **names) {
    int i;
    int base_len = strlen(path) - strlen(".raw");
    float3 o, d; /* origin, spacing */
    o.x = o.y = o.z = 0.0;
    d = get_spacing(D, G);

    fprintf(f, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(f, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", 1 + G.z, 1 + G.y, 1 + G.x);
    fprintf(f, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(f, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", o.x, o.y, o.z);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", d.x, d.y, d.z);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "     </Geometry>\n");
    for(i = 0; i < n; ++i) {
        fprintf(f, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", names[i]);
        fprintf(f, "       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"Binary\">\n", G.z, G.y, G.x);
        fprintf(f, "        %.*s.%s.raw\n", base_len, path, names[i]);
        fprintf(f, "       </DataItem>\n");
        fprintf(f, "     </Attribute>\n");
    }
    fprintf(f, "   </Grid>\n");
}

static int eq(const char *a, const char *b) { return !strcmp(a, b); }
static void xsuffix(const char *i, /**/ char *o) {
    /* replace suffix .raw by .xmf */
    char s[] = ".raw";
    int n;
    n = strlen(i) - strlen(s);
    if (n >= 0 && eq(i + n, s)) {strncpy(o, i, n); o[n] = '\0';}
    else                        strcpy (o, i);
    strcat(o, ".xmf");
}
static void basename(const char *i, /**/ char *o) {
    const char *p;
    p = i;
    while (*i != '\0') if (*i++ == '/') p = i;
    strcpy(o, p);
}

void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names) {
    char w[BUFSIZ];
    FILE *f;

    xsuffix(path, /**/ w);
    UC(efopen(w, "w", /**/ &f));
    header(f);

    basename(path, /**/ w);
    grid(domainSize, gridSize, f, w, ncomp, names);
    epilogue(f);
    UC(efclose(f));
}





#include <mpi.h>
#include <stdio.h>
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
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names);
void raw_write(int3 N, MPI_Comm, const char *path, int ncomp, const float **data, const char **names);

static int3 get_total_size(int3 L, MPI_Comm cart) {
    enum {X, Y, Z, D};
    int dims[3], periods[D], coords[D];
    int3 G;
    MC(m::Cart_get(cart, D, dims, periods, coords));
    G.x = L.x * dims[X];
    G.y = L.y * dims[Y];
    G.z = L.z * dims[Z];
    return G;
}

void grid_write(int3 N, int3 L, MPI_Comm cart, const char *path, int ncmp, const float **data, const char **names) {
    int3 domainSize, gridSize;

    domainSize = get_total_size(L, cart);
    gridSize   = get_total_size(N, cart);

    UC(raw_write(N, cart, path, ncmp, data, names));
    if (m::is_master(cart))
        UC(xmf_write(domainSize, gridSize, path, ncmp, names));
}







// Grouped editable compilation unit for text and diagnostic I/O.


// Grouped editable compilation unit for text and communication I/O.



#ifndef UDX_IO_COM_COMMON
#define UDX_IO_COM_COMMON

#include <stdio.h>
#include <vector_types.h>
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
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

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
void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
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

struct WriteFile;

int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*); // <1>
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*); // <2>

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);

struct Coords;
struct float3;
void io_com_dump(MPI_Comm, const Coords*, const char *name, long id, int n, const int *ii, const float3 *rr);

enum {MAX_CHAR_PER_LINE = 128};

#define BASE DUMP_BASE "/diag/com"

static int swrite(const Coords *coords, int n, const int *ii, const float3 *rr, /**/ char *s) {
    int i, id, c, start;
    float3 r, v, rg;
    for (i = start = 0; i < n; ++i) {
        id = ii[i];
        r  = rr[i];
        local2global(coords, r, /**/ &rg);
        c = sprintf(s + start, "%d %g %g %g %g %g %g\n", id, rg.x, rg.y, rg.z, v.x, v.y, v.z);
        if (c >= MAX_CHAR_PER_LINE)
            ERR("buffer too small : %d / %d", c, MAX_CHAR_PER_LINE);
        start += c;
    }
    return start;
}

static void io_com_write_mpi(MPI_Comm comm, const char *fname, long n, const char *data) {
    WriteFile *f;
    UC(write_file_open(comm, fname, &f));
    UC(write_all(comm, data, n, f));
    UC(write_file_close(f));
}

static void io_com_gen_fname(const char *name, long id, char *fname) {
    sprintf(fname, BASE "/%s.%04ld.txt", name, id);
}

void io_com_dump(MPI_Comm comm, const Coords *coords, const char *name, long id, int n, const int *ii, const float3 *rr) {
    char fname[FILENAME_MAX] = {0}, *data;
    long nchar = 0;

    EMALLOC(MAX_CHAR_PER_LINE * n, &data);

    if (m::is_master(comm))
        UC(os_mkdir(BASE));
    MC(m::Barrier(comm));

    io_com_gen_fname(name, id, fname);

    UC(nchar = swrite(coords, n, ii, rr, /**/ data));
    io_com_write_mpi(comm, fname, nchar, data);

    EFREE(data);
}

#undef BASE
#endif





#include <stdio.h>

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_IO_DIAG_COMMON_CUH
#define UDX_IO_DIAG_COMMON_CUH
#include "types.cuh"
#endif
#endif

void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
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

struct Particle;
struct Force;
struct TxtRead;

void txt_write_pp(long n, const Particle*, const char *path);
void txt_write_pp_ff(long n, const Particle*, const Force*, const char *path);

void txt_read_pp(const char *path, TxtRead **);
void txt_read_pp_ff(const char *path, TxtRead **);
void txt_read_ff(const char *path, TxtRead **);
void txt_read_fin(TxtRead*);

int txt_read_get_n(const TxtRead *);
const Particle* txt_read_get_pp(const TxtRead *);
const Force*    txt_read_get_ff(const TxtRead *);

struct TxtRead {
    int n;
    Particle *pp;
    Force *ff;
};
#define VFRMT "%16.10e %16.10e %16.10e"

void txt_write_pp(long n, const Particle *pp, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    const float *r, *v;
    int rcode;

    UC(efopen(name, "w", /**/ &f));
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        v = pp[i].v;
        rcode = fprintf(f, VFRMT " " VFRMT "\n",
                        r[X], r[Y], r[Z], v[X], v[Y], v[Z]);
        if (rcode < 0) ERR("fprintf to file '%s' failed", name);
    }
    UC(efclose(f));
    msg_print("dump %ld particles to '%s'", n, name);
}

void txt_write_pp_ff(long n, const Particle *pp, const Force *ff, const char *name) {
    enum {X, Y, Z};
    long i;
    FILE *f;
    const float *r, *v, *a;
    int rcode;

    UC(efopen(name, "w", /**/ &f));
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        v = pp[i].v;
        a = ff[i].f;
        rcode = fprintf(f, VFRMT " " VFRMT " " VFRMT "\n",
                        r[X], r[Y], r[Z], v[X], v[Y], v[Z], a[X], a[Y], a[Z]);
        if (rcode < 0) ERR("fprintf to file '%s' failed", name);
    }
    UC(efclose(f));
    msg_print("dump %ld particles to '%s'", n, name);
}

#undef VFRMT
#define VFRMT "%f %f %f"

static void ini(TxtRead **pq) {
    TxtRead *q;
    EMALLOC(1, &q);
    q->pp = NULL;
    q->ff = NULL;
    q->n  = -1;
    *pq = q;
}

static int get_num_lines(FILE *f) {
    int n;
    char c;
    n = 0;
    while (EOF != (c = fgetc(f)))
        if (c == '\n') ++n;
    return n;
}

void txt_read_pp(const char *name, TxtRead **pq) {
    enum {X, Y, Z};
    TxtRead *q;
    FILE *f;
    Particle p;
    int i;
    UC(ini(&q));
    q->ff = NULL;
    UC(efopen(name, "r", /**/ &f));
    q->n = get_num_lines(f);
    EMALLOC(q->n, &q->pp);
    rewind(f);
    i = 0;
    while (6 == fscanf(f, VFRMT " " VFRMT "\n",
                       p.r + X, p.r + Y, p.r + Z,
                       p.v + X, p.v + Y, p.v + Z)) {
        q->pp[i++] = p;
    }
    UC(efclose(f));
    *pq = q;
}

void txt_read_pp_ff(const char *name, TxtRead **pq) {
    enum {X, Y, Z};
    TxtRead *q;
    FILE *f;
    Particle p;
    Force fo;
    int i;
    UC(ini(&q));
    UC(efopen(name, "r", /**/ &f));
    q->n = get_num_lines(f);
    EMALLOC(q->n, &q->pp);
    EMALLOC(q->n, &q->ff);
    rewind(f);
    i = 0;
    while (9 == fscanf(f, VFRMT " " VFRMT " " VFRMT "\n",
                       p.r + X, p.r + Y, p.r + Z,
                       p.v + X, p.v + Y, p.v + Z,
                       fo.f + X, fo.f + Y, fo.f + Z)) {
        q->pp[i] = p;
        q->ff[i] = fo;
        ++i;
    }
    UC(efclose(f));
    *pq = q;
}

void txt_read_ff(const char *name, TxtRead **pq) {
    enum {X, Y, Z};
    TxtRead *q;
    FILE *f;
    Force fo;
    int i;
    UC(ini(&q));
    UC(efopen(name, "r", /**/ &f));
    q->n = get_num_lines(f);
    EMALLOC(q->n, &q->ff);
    rewind(f);
    i = 0;
    while (3 == fscanf(f, VFRMT "\n",
                       fo.f + X, fo.f + Y, fo.f + Z)) {
        q->ff[i] = fo;
        ++i;
    }
    UC(efclose(f));
    *pq = q;
}


void txt_read_fin(TxtRead *q) {
    EFREE(q->pp);
    EFREE(q->ff);
    EFREE(q);
}

int txt_read_get_n(const TxtRead *q) {return q->n;}
const Particle* txt_read_get_pp(const TxtRead *q) { return q->pp; }
const Force*    txt_read_get_ff(const TxtRead *q) { return q->ff; }

#undef VFRMT





// Grouped editable compilation unit for diagnostic output.



#include <stdio.h>
#include <string.h>
#include <vector_types.h>
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
#include "diagnostic.cuh"
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
struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);

struct DiagMesh;
struct MeshRead;

void diag_mesh_ini(const char *path, MeshRead*, DiagMesh**);
void diag_mesh_fin(DiagMesh*);
void diag_mesh_apply(DiagMesh*, MPI_Comm, float time, int nc, Particle*);
struct DiagMesh {
    char path[FILENAME_MAX];
    int nv, nt;
    int4 *tt;
};
void diag_mesh_ini(const char *path, MeshRead *mesh, DiagMesh **pq) {
    FILE *f;
    DiagMesh *q;
    EMALLOC(1, &q);
    *pq = q;
    q->nv = mesh_read_get_nv(mesh);
    q->nt = mesh_read_get_nt(mesh);
    EMALLOC(q->nt, &q->tt);
    EMEMCPY(q->nt, mesh_read_get_tri(mesh), q->tt);
    snprintf(q->path, FILENAME_MAX, "%s/%s", DUMP_BASE, path);
    UC(efopen(q->path, "w", /**/ &f));
    UC(efclose(f));
    msg_print("DiagMesh: %s", q->path);
}

void diag_mesh_fin(DiagMesh *q) {
    EFREE(q->tt);
    EFREE(q);
}

void diag_mesh_apply(DiagMesh*, MPI_Comm, float, int, Particle*) {

}





#include <stdio.h>
#include <string.h>
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
#include "diagnostic.cuh"
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
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

struct Particle;
struct DiagPart;

void diag_part_ini(const char *path, /**/ DiagPart**);
void diag_part_fin(DiagPart*);
void diag_part_apply(DiagPart*, MPI_Comm, float time, int n, const Particle*);
enum {STAMP_GOOD = 123456};
struct DiagPart {
    int stamp;
    char path[FILENAME_MAX];
};
void diag_part_ini(const char *path, /**/ DiagPart **pq) {
    FILE *f;
    DiagPart *q;
    EMALLOC(1, &q);
    snprintf(q->path, FILENAME_MAX, "%s/%s", DUMP_BASE, path);
    UC(efopen(q->path, "w", /**/ &f));
    UC(efclose(f));
    msg_print("DiagPart: %s", q->path);
    q->stamp = STAMP_GOOD;
    *pq = q;
}
void diag_part_fin(DiagPart *q) { EFREE(q); }

static float sq(float x) { return x*x; }

#ifndef UDX_SCHEME_IO_REDUCE_COMMON
#define UDX_SCHEME_IO_REDUCE_COMMON
static void reduce(MPI_Comm comm, const void *sendbuf0, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    int root = 0;
    const void *sendbuf = (m::is_master(comm) ? MPI_IN_PLACE : sendbuf0);
    MC(m::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm));
}
#endif /* UDX_SCHEME_IO_REDUCE_COMMON */
static void sum3(MPI_Comm comm, double *v) {
    reduce(comm, v, v, 3, MPI_DOUBLE, MPI_SUM);
}

static void sum_d(MPI_Comm comm, double *v) {
    reduce(comm, v, v, 1, MPI_DOUBLE, MPI_SUM);
}

static void max_d(MPI_Comm comm, double *v) {
    reduce(comm, v, v, 1, MPI_DOUBLE, MPI_MAX);
}

static void sum_i(MPI_Comm comm, int *v) {
    reduce(comm, v, v, 1, MPI_INT, MPI_SUM);
}

void diag_part_apply(DiagPart *q, MPI_Comm comm, float time, int n, const Particle *pp) {
    if (time < 0) ERR("time = %g < 0", time);
    if (q->stamp != STAMP_GOOD) ERR("the DiagPart object was not initialized");
    enum {X, Y, Z, D};
    int i, c;
    double k, km, ke; /* particle, total, and maximum kinetic energies */
    double kbt = 0;
    FILE *f = NULL;
    double v[D] = {0};
    for (i = 0; i < n; ++i) for (c = 0; c < D; ++c) v[c] += pp[i].v[c];
    UC(sum3(comm, v));

    ke = km = 0;
    for (i = 0; i < n; ++i) {
        k = sq(pp[i].v[X]) + sq(pp[i].v[Y]) + sq(pp[i].v[Z]);
        ke += k;
        if (k > km) km = k;
    }

    UC(sum_d(comm, &ke); max_d(comm, &km));
    UC(sum_i(comm, &n));

    if (m::is_master(comm)) {
        if (n) kbt = 0.5 * ke / (n * D / 2.0);
        UC(efopen(q->path, "a", /**/ &f));
        msg_print("% .1e % .1e [% .1e % .1e % .1e] % .1e", time, kbt, v[X], v[Y], v[Z], km);
        fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", time, kbt, v[X], v[Y], v[Z], km);
        UC(efclose(f));
    }
}







// Grouped editable compilation unit for mesh, rigid, and restart output.



#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

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
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#endif

struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
struct Particle;
struct Vectors;
struct Coords;

void vectors_float_ini(int n, const float*, /**/ Vectors**); // <1>
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**); // <2>
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*, /**/ Vectors**); // <3>
void vectors_postions_center_ini(const Coords*, int n, const Particle*, /**/ Vectors**); // <4>
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**); // <5>
void vectors_zero_ini(int n, /**/ Vectors**); // <6>

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i, /**/ float r[3]); // <7>

struct WriteFile;

int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*); // <1>
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*); // <2>

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);

struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

#ifndef UDX_SIM_MESH_WRITE_COMMON
#define UDX_SIM_MESH_WRITE_COMMON
enum {MESH_WRITE_EDGE, MESH_WRITE_CENTER};
#endif

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory, /**/ MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
struct MeshWrite {
    int4 *tt;
    int nv, nt;
    char directory[FILENAME_MAX];
    int shift_type;
};

#ifndef UDX_IO_OUTPUT_COMMON_UTIL
#define UDX_IO_OUTPUT_COMMON_UTIL
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
#endif

static void mkdir0(const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

#ifndef UDX_IO_OUTPUT_MKDIR_COMMON
#define UDX_IO_OUTPUT_MKDIR_COMMON
static void mkdir(MPI_Comm comm, const char *p, const char *s) {
    if (m::is_master(comm)) UC(mkdir0(p, s));
}
#endif
#define PATTERN "%s/%s/%05d.ply"
enum { NVP = 3 /* number of vertices per face */ };

static void ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    int i;
    MeshWrite *q;
    EMALLOC(1, &q);

    q->nv = nv; q->nt = nt;
    cpy(q->directory, directory);
    UC(mkdir(comm, DUMP_BASE, directory));
    EMALLOC(nt, &q->tt);
    for (i = 0; i < nt; i++)
        q->tt[i] = tt[i];
    q->shift_type = shifttype;
    *pq = q;
}

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    UC(ini(comm, shifttype, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead *cell, const char *directory, /**/ MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tt = mesh_read_get_tri(cell);
    UC(ini(comm, shifttype, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_fin(MeshWrite *q) {
    EFREE(q->tt);
    EFREE(q);
}

static void header(MPI_Comm comm, int nc0, int nv, int nt, WriteFile *f) {
    int nc; /* total number of cells */
    int sz = 0;
    char s[BUFSIZ];
    UC(write_reduce(comm, nc0, &nc));
    if (m::is_master(comm))
        sz = snprintf(s, BUFSIZ - 1,
                     "ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex %d\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float u\nproperty float v\nproperty float w\n"
                     "element face  %d\n"
                     "property list int int vertex_index\n"
                     "end_header\n", nv*nc, nt*nc);
    write_master(comm, s, sz, f);
}

static void vert(MPI_Comm cart, int n, const Vectors *pos, const Vectors *vel, WriteFile *f) {
    float *D, *r, *v;
    int i, j, m;
    m = (3 + 3) * n; /* pos[3] + vel[3] */
    EMALLOC(m, &D);
    for (i = j = 0; i < n; i++) {
        r = &D[j]; j += 3;
        v = &D[j]; j += 3;
        UC(vectors_get(pos, i, /**/ r));
        UC(vectors_get(vel, i, /**/ v));
    }
    UC(write_all(cart, D, m*sizeof(D[0]), f));
    EFREE(D);
}

static void wfaces0(MPI_Comm comm, int *buf, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {
    /* write faces */
    int c, t, b;  /* cell, triangle, buffer index */
    int n, shift;
    int4 tri;
    n = nc * nv;
    UC(write_shift_indices(comm, n, &shift));
    for (b = c = 0; c < nc; ++c)
        for (t = 0; t < nt; ++t) {
            tri = faces[t];
            buf[b++] = NVP;
            buf[b++] = shift + nv*c + tri.x;
            buf[b++] = shift + nv*c + tri.y;
            buf[b++] = shift + nv*c + tri.z;
        }
    UC(write_all(comm, buf, b*sizeof(buf[0]), f));
}

static void wfaces(MPI_Comm cart, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {
    int *buf; /* buffer for faces */
    int sz;
    sz = (1 + NVP) * nc * nt;
    EMALLOC(sz, &buf);
    UC(wfaces0(cart, buf, faces, nc, nv, nt, f));
    EFREE(buf);
}

static void mesh_write(MPI_Comm comm, int nc, int nv, int nt,
                        const Vectors *pos, const Vectors *vel, const int4 *faces,
                        WriteFile *f) {
    int n;
    n = nc * nv;
    UC(header(comm, nc, nv, nt, f));
    UC(vert(comm, n, pos, vel, f));
    UC(wfaces(comm, faces, nc, nv, nt, f));
}

void mesh_write_particles(MeshWrite *q, MPI_Comm comm, const Coords *coords, int nc, const Particle *pp, int id) {
    int nv, nt, n;
    const int4 *tt;
    Vectors *pos, *vel;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX] = {'\0'} ;

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    n = nv * nc;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("sprintf failed");
    UC(write_file_open(comm, path, /**/ &f));
    switch (q->shift_type) {
    case MESH_WRITE_EDGE:
        vectors_postions_edge_ini(coords, n, pp, /**/ &pos);
        break;
    case MESH_WRITE_CENTER:
        vectors_postions_center_ini(coords, n, pp, /**/ &pos);
        break;
    default:
        ERR("unkown q->type: %d", q->shift_type);
    }
    vectors_velocities_ini(n, pp, /**/ &vel);
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    vectors_fin(pos);
    vectors_fin(vel);
    UC(write_file_close(f));
}

void mesh_write_vectors(MeshWrite *q, MPI_Comm comm, int nc, Vectors *pos, Vectors *vel, int id) {
    int nv, nt;
    const int4 *tt;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX] = {'\0'};

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("snprintf failed");
    UC(write_file_open(comm, path, /**/ &f));
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    UC(write_file_close(f));
}




// Grouped editable compilation unit for rigid output and restart I/O.



#include <stdio.h>
#include <string.h>
#include <mpi.h>

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_IO_TYPE_COMMON
#define UDX_IO_TYPE_COMMON
#include "types.cuh"
#endif
#endif

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
void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
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

struct WriteFile;

int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*); // <1>
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*); // <2>

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);

struct Coords;
struct Rigid;

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Rigid *ss);

struct IoRig {
    char mode[16]; // fopen mode
};
#define BASE DUMP_BASE "/diag/rig"

static int swrite_v(const float v[3], char *buf) {
    enum {X, Y, Z};
    return sprintf(buf, "%+.6e %+.6e %+.6e ", v[X], v[Y], v[Z]); // 3 * 13 + 3 = 42
}

static int swrite(const Coords *c, int ns, const Rigid *ss, int maxn, char *buf) {
    enum {X, Y, Z, D};
    int i, nchar;
    float com[D];
    const Rigid *s;

    for (i = nchar = 0; i < ns; ++i) {
        s = &ss[i];

        // shift to global coordinates
        com[X] = xl2xg(c, s->com[X]);
        com[Y] = yl2yg(c, s->com[Y]);
        com[Z] = zl2zg(c, s->com[Z]);

        nchar += sprintf(buf + nchar, "%04d ", (int) s->id); // 5
        nchar += swrite_v(com,   buf + nchar);
        nchar += swrite_v(s->v,  buf + nchar);
        nchar += swrite_v(s->om, buf + nchar);
        nchar += swrite_v(s->fo, buf + nchar);
        nchar += swrite_v(s->to, buf + nchar);
        nchar += swrite_v(s->e0, buf + nchar);
        nchar += swrite_v(s->e1, buf + nchar);
        nchar += swrite_v(s->e2, buf + nchar);
        nchar += sprintf(buf + nchar, "\n"); // 1
        if (nchar > maxn)
            ERR("exceed buffer capacity: [%d / %d]", nchar, maxn);
    }
    return nchar;
}

static void io_rig_write_mpi(MPI_Comm comm, const char *fname, long n, const char *data) {
    WriteFile *f;
    UC(write_file_open(comm, fname, &f));
    UC(write_all(comm, data, n, f));
    UC(write_file_close(f));
}

static void io_rig_gen_fname(const char *name, long id, char *fname) {
    sprintf(fname, BASE "/%s.%04ld.txt", name, id);
}

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Rigid *ss) {
    char fname[FILENAME_MAX], *data;
    int maxnc, nchar = 0;

    enum {
        NC_ID = 5,
        NC_END = 1,
        NC_v3 = 42,
        NC_SECURITY = 1,
        N_v3 = 8
    };

    maxnc = ns * (NC_ID + NC_END + NC_v3 * N_v3 + NC_SECURITY);

    EMALLOC(maxnc, &data);

    if (m::is_master(comm))
        UC(os_mkdir(BASE));
    MC(m::Barrier(comm));

    io_rig_gen_fname(name, id, fname);
    nchar = swrite(c, ns, ss, maxnc, data);
    io_rig_write_mpi(comm, fname, nchar, data);

    EFREE(data);
}

#undef BASE





#include <stdio.h>
#include <string.h>
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

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
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

struct Coords;
struct Particle;
struct Rigid;

#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
#endif

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
/*
   pattern :
   base/code/id.ext
   base depends on read/write
*/

#define PATTERN "%s/%s/%s"

// buff size
enum { BS = 1024 };

// check fprintf (BS-1 for \0 character)
#define CSPR(a) do {                                        \
        int check = a;                                      \
        if (check < 0 || check >= BS-1)                     \
        ERR("Buffer too small to handle this format\n");    \
    } while (0)

static void id2code(const int id, char *code) {
    switch (id) {
    case RESTART_TEMPL:
        CSPR(sprintf(code, "templ"));
        break;
    case RESTART_FINAL:
        CSPR(sprintf(code, "final"));
        break;
    default:
        CSPR(sprintf(code, "%05d", id));
        break;
    }
}

static void gen_base_name(const char *pattern, const char *base, const char *code, const int id, /**/ char *name) {
    char idcode[BS] = {0};
    id2code(id, /**/ idcode);
    CSPR(sprintf(name, pattern, base, code, idcode));
}

static void gen_base_name_dump(const char *base, const char *code, const int id, /**/ char *name) {
    gen_base_name(PATTERN, base, code, id, name);
}

static void gen_base_name_read(const char *base, const char *code, const int id, /**/ char *name) {
    gen_base_name(PATTERN ".bop", base, code, id, name);
}

template <typename T>
static void write(MPI_Comm comm, const char *base, const char *code, int id, long n, const T *tt, BopType type, int nvars, const char *vars) {
    char name[BS];
    BopData *bop;
    T *tt_bop;
    gen_base_name_dump(base, code, id, /**/ name);

    BPC(bop_ini(&bop));
    BPC(bop_set_n(n, bop));
    BPC(bop_set_type(type, bop));
    BPC(bop_set_vars(nvars, vars, bop));
    BPC(bop_alloc(bop));

    tt_bop = (T*) bop_get_data(bop);
    memcpy(tt_bop, tt, n * sizeof(T));

    BPC(bop_write_header(comm, name, bop));
    BPC(bop_write_values(comm, name, bop));

    BPC(bop_fin(bop));
}

template <typename T>
static void read(MPI_Comm comm, const char *base, const char *code, const int id, int *n, T *tt) {
    long np = 0;
    char hname[BS], dname[BS];
    BopData *bop;
    const T *tt_bop;
    gen_base_name_read(base, code, id, /**/ hname);
    msg_print("reading '%s'", hname);

    BPC(bop_ini(&bop));
    BPC(bop_read_header(comm, hname, bop, dname));
    BPC(bop_alloc(bop));
    BPC(bop_read_values(comm, dname, bop));

    BPC(bop_get_n(bop, &np));
    tt_bop = (const T*) bop_get_data(bop);
    memcpy(tt, tt_bop, np * sizeof(T));
    *n = np;

    BPC(bop_fin(bop));
}

void restart_write_pp(MPI_Comm comm, const char *base, const char *code, int id, long n, const Particle *pp) {
    write(comm, base, code, id, n, pp, BopFLOAT, 6, "x y z vx vy vz");
}

void restart_read_pp(MPI_Comm comm, const char *base, const char *code, int id, int *n, Particle *pp) {
    read(comm, base, code, id, n, pp);
}

void restart_write_ii(MPI_Comm comm, const char *base, const char *code, int id, long n, const int *ii) {
    write(comm, base, code, id, n, ii, BopINT, 1, "i");
}

void restart_read_ii(MPI_Comm comm, const char *base, const char *code, int id, int *n, int *ii) {
    read(comm, base, code, id, n, ii);
}

void restart_write_ss(MPI_Comm comm, const char *base, const char *code, int id, long n, const Rigid *ss) {
    static const char *vars =
        "Ixx Ixy Ixz Iyy Iyz Izz "
        "mass x y z vx vy vz omx omy omz "
        "e0x e0y e0z e1x e1y e1z e2x e2y e2z "
        "fx fy fz tx ty tz i";
    enum {NVARS=32};
    write(comm, base, code, id, n, ss, BopFLOAT, NVARS, vars);
}

void restart_read_ss(MPI_Comm comm, const char *base, const char *code, int id, int *n, Rigid *ss) {
    read(comm, base, code, id, n, ss);
}

void restart_write_stream_one_node(MPI_Comm comm, const char *base, const char *code, int id, const void *data, StreamWriter sw) {
    char name[BS];
    FILE *f;
    if (!m::is_master(comm)) return;
    gen_base_name(PATTERN ".dat", base, code, id, name);
    UC(efopen(name, "w", &f));
    UC(sw(data, f));
    UC(efclose(f));
}

void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader sr, void *data) {
    char name[BS];
    FILE *f;
    gen_base_name(PATTERN ".dat", base, code, id, name);
    msg_print("reading '%s'", name);
    UC(efopen(name, "r", &f));
    UC(sr(f, data));
    UC(efclose(f));
}

#undef PATTERN





// Grouped editable compilation unit for mesh reading and edge topology.



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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

void edg_ini(int md, int nv,                     int *hx);
void edg_set(int md, int i, int j, int val, /**/ int *hx,       int *hy); // <1>
int  edg_get(int md, int i, int j,         const int *hx, const int *hy); // <2>
int  edg_valid(int md, int i, int j,         const int *hx); // <3>

struct Edg {
    int md, nv;
    int *hx, *hy;
};
void e_ini(int md, int nv, /**/ Edg **pq) {
    Edg *q;
    EMALLOC(1, &q);
    EMALLOC(md*nv, &q->hx);
    EMALLOC(md*nv, &q->hy);
    edg_ini(md, nv, /**/ q->hx);
    q->md = md; q->nv = nv;
    *pq = q;
}

void e_fin(Edg *q) {
    EFREE(q->hx);
    EFREE(q->hy);
    EFREE(q);
}

void e_set(Edg *q, int i, int j, int val) {
    int md, *hx, *hy;
    md = q->md; hx = q->hx; hy = q->hy;
    UC(edg_set(md, i, j, val, hx, hy));
}

int e_get(Edg *q, int i, int j) {
    int md, ans;
    const int *hx, *hy;
    md = q->md; hx = q->hx; hy = q->hy;
    UC(ans = edg_get(md, i, j, hx, hy));
    return ans;
}

int e_valid(Edg *q, int i, int j) {
    int md;
    const int *hx;
    md = q->md; hx = q->hx;
    return edg_valid(md, i, j, hx);
}





#include <stdio.h>
#include <string.h>
#include <vector_types.h>

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

void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
struct Edg;

void e_ini(int md, int nv, /**/ Edg**);
void e_fin(Edg*);

void e_set(Edg*, int i, int j, int val);
int  e_get(Edg*, int i, int j);
int  e_valid(Edg*, int i, int j);

struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);

struct MeshRead {
    int nv, nt, nd; /* vertices, triangles, dihidrals */
    float *rr; /* [x y z] [x y z] */
    int4  *tt; /* triangles: [v0 v1 v2 dummy] */
    int4  *dd; /* dihidrals: [v0 (v1 v2) v3]  */

    int dd_computed;
};
#define SIZE 1024
static int emptyp(char *s) {
    for (;;) {
        if      (s[0] == '\0')                return 1;
        else if (s[0] != ' ' && s[0] != '\t') return 0;
        else s++;
    }
}
static int commentp(char *s) {
    return s[0] != '\0' && s[0] == '#';
}
static void unend(char *s) { /* rm \n */
    if (s[0] == '\0') return;
    while (s[1] != '\0') s++;
    if (s[0] == '\n') s[0] = '\0';
}
enum {OK, ERR};
static int line(FILE *f, /**/ char *s) { /* read line */
    for (;;) {
        if (fgets(s, SIZE, f) == NULL)  return ERR;
        unend(s);
        if (!emptyp(s) && !commentp(s)) break;
    }
    return OK;
}

static int header(FILE *f, char *s) {
    char key[SIZE];
    if (line(f, /**/ s) == ERR) return 0;
    sscanf(s, "%s", key);
    return same_str(key, "OFF");
}

static int sizes(FILE *f, char *s, /**/ int *nv, int *nt) {
    int r;
    if (line(f, /**/ s) == ERR) return 0;
    r = sscanf(s, "%d %d %*d", nv, nt);
    return r == 2;
}
static int vert(FILE *f, char *s, int n, float *rr) {
    enum {X, Y, Z};
    int i;
    float *r;
    for (i = 0; i < n; i++) {
        if (line(f, /**/ s) == ERR) return 0;
        r = &rr[3*i];
        if (sscanf(s, "%f %f %f", &r[X], &r[Y], &r[Z]) != 3) return 0;
    }
    return 1;
}
static int good(int4 *t, int n) {
    int x, y, z;
    x = t->x; y = t->y; z = t->z;
    return
        (0 <= x  && x < n) &&
        (0 <= y  && y < n) &&
        (0 <= z  && z < n) &&
        (x != y  && y != z);
}
static int tri(FILE *f, char *s, int nt, int nv, int4 *tt) {
    int i;
    int4 *t;
    int nvpt; /* number of vertices per face (triangle) */
    for (i = 0; i < nt; i++) {
        if (line(f, /**/ s) == ERR) return 0;
        t = &tt[i];
        if (sscanf(s, "%d %d %d %d", &nvpt, &t->x, &t->y, &t->z) != 4) {
            msg_print("expecting triangle:'%s'", s);
            return 0;
        }
        if (nvpt != 3) {
            msg_print("expecting triangle: '%s'", s);
            return 0;
        }
        if (!good(t, nv)) {
            msg_print("wrong triangle: '%s'", s);
            msg_print("nv = %d, nt = %d", nv, nt);
            return 0;
        }
    }
    return 1;
}
static void read_off(FILE *f, const char *path, /**/ MeshRead *q) {
    int nv, nt;
    char s[SIZE];
    float *rr;
    int4  *tt;
    if (!header(f, s)) {
        msg_print("not an OFF file: '%s'", s);
        ERR("failed to read: '%s'", path);
    }
    if (!sizes(f, s, &nv, &nt)) {
        msg_print("wrong sizes: '%s'", s);
        ERR("failed to read: '%s'", path);
    }
    EMALLOC(3*nv, &rr);
    EMALLOC(  nt, &tt);
    if (!vert(f, s, nv, /**/ rr))
        ERR("failed to read vertices: '%s'", path);
    if (!tri(f, s, nt, nv, /**/ tt))
        ERR("failed to read triangles: '%s'", path);
    q->nv = nv; q->nt = nt;
    q->rr = rr; q->tt = tt;
}
static bool prop(const char *pname, const char *str) {
    const int l1 = strlen(pname);
    const int l2 = strlen(str);

    if (l1 > l2) return false;
    for (int i = 0; i < l1; ++i) if (pname[i] != str[i]) return false;
    return true;
}

static void read_ply(FILE *f, const char *fname, MeshRead *q) {
    int nv, nt;
    int4 *tt;
    float *rr;
    int l = 0;
#define BUFSIZE 256 // max number of chars per line
#define MAXLINES 64 // max number of line for header

    // https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html#Stringizing
#define xstr(s) str(s)
#define str(s) #s

    nv = nt = -1;
    while (l++ < MAXLINES) {
        char cbuf[BUFSIZE + 1] = {0}; // + 1 for \0

        const int checker = fscanf(f, " %[^\n]" xstr(BUFSIZE) "c", cbuf);
        if (checker != 1)
            ERR("Fail to read '%s'", fname);

        int ibuf;
        if    (sscanf(cbuf, "element vertex %d", &ibuf) == 1) nv = ibuf;
        else if (sscanf(cbuf, "element face %d", &ibuf) == 1) nt = ibuf;
        else if (prop("end_header", cbuf)) break;
    }
    if (l >= MAXLINES || nt == -1 || nv == -1)
        ERR("Fail to read '%s': did not catch end_header", fname);
    EMALLOC(  nt, &tt);
    EMALLOC(3*nv, &rr);
    for (int i = 0; i < nv; ++i)
        fscanf(f, "%f %f %f\n",
               rr + 3*i + 0, rr + 3*i + 1, rr + 3*i + 2);
    int4 t; t.z = 0;
    for (int i = 0; i < nt; ++i) {
        fscanf(f, "%*d %d %d %d\n", &t.x, &t.y, &t.z);
        tt[i] = t;
    }
    q->nv = nv; q->nt = nt;
    q->rr = rr; q->tt = tt;
}
struct Q { int4 *d; }; /* queue */

static void q_push(Q *q, int i0, int i1, int i2, int i3) {
    int4 *d;
    d = q->d;
    d->x = i0; d->y = i1; d->z = i2; d->w = i3;
    (q->d)++;
}

static void reg(int i1, int i2, Edg *nxt, Edg *seen, /**/ Q *q) {
    int i0, i3;
    if (e_valid(seen, i1, i2)) return;
    if (e_valid(seen, i2, i1)) return;
    i0 = e_get(nxt, i1, i2);
    i3 = e_get(nxt, i2, i1); /* previous */
    q_push(q, i0, i1, i2, i3);
    UC(e_set(seen, i1, i2, 1));
}

static void ini_dd0(int nt, const int4 *tt, int nv, int md, /**/ int4 *dd) {
    Q q;
    Edg *nxt, *seen;
    int i, i0, i1, i2;
    q.d = dd;

    UC(e_ini(md, nv, &nxt));
    UC(e_ini(md, nv, &seen));

    for (i = 0; i < nt; i++) { /* save who is next for every edge */
        i0 = tt[i].x; i1 = tt[i].y; i2 = tt[i].z;
        e_set(nxt, i0, i1, i2);
        e_set(nxt, i1, i2, i0);
        e_set(nxt, i2, i0, i1);
    }

    for (i = 0; i < nt; i++) { /* register dihidrals */
        i0 = tt[i].x; i1 = tt[i].y; i2 = tt[i].z;
        reg(i0, i1, nxt, /**/ seen, &q);
        reg(i1, i2, nxt, /**/ seen, &q);
        reg(i2, i0, nxt, /**/ seen, &q);
    }

    UC(e_fin(nxt); e_fin(seen));
}

static void ini_dd(/**/ MeshRead *q) {
    int nt, nv, nd, md;
    const int4 *tt;
    nt = mesh_read_get_nt(q);
    nv = mesh_read_get_nv(q);
    nd = mesh_read_get_ne(q);
    md = mesh_read_get_md(q);
    tt = mesh_read_get_tri(q);
    q->nd = nd;
    EMALLOC(nd, &q->dd);
    UC(ini_dd0(nt, tt, nv, md, /**/ q->dd));
}

void mesh_read_ini_off(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    EMALLOC(1, &q);
    UC(efopen(path, "r", /**/ &f));
    UC(read_off(f, path, /**/ q));
    UC(efclose(f));
    msg_print("read off '%s'", path);
    q->dd_computed = 0;
    q->dd = NULL;
    *pq = q;
}

void mesh_read_ini_ply(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    EMALLOC(1, &q);
    UC(efopen(path, "r", /**/ &f));
    UC(read_ply(f, path, /**/ q));
    UC(efclose(f));
    msg_print("read ply '%s'", path);
    q->dd = NULL;
    *pq = q;
}

void mesh_read_fin(MeshRead* q) {
    EFREE(q->rr);
    EFREE(q->tt);
    if (q->dd) EFREE(q->dd);
    EFREE(q);
}

int mesh_read_get_nv(const MeshRead *q) { return q->nv; }
int mesh_read_get_nt(const MeshRead *q) { return q->nt; }
int mesh_read_get_ne(const MeshRead *q) {
    int nt;
    nt = q->nt;
    if (nt % 2 != 0) ERR("nt=%d % 2 != 0", nt);
    return 3 * nt / 2;
}

static int amax(int *a, int n) {
    int i, m;
    if (n <= 0) ERR("amax called with size: %d\n", n);
    m = a[0];
    for (i = 1; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}

int mesh_read_get_md(const MeshRead *q) {
    int *d;
    int i, m, nt, nv;
    int x, y, z;
    int4 *tt;
    nt = q->nt; nv = q->nv; tt = q->tt;

    if (nv == 0) {
        msg_print("mesh_get_md called for nv = 0");
        return 0;
    }
    EMALLOC(nv, &d);
    for (i = 0; i < nv; i++) d[i] = 0;
    for (i = 0; i < nt; i++) {
        x = tt[i].x; y = tt[i].y; z = tt[i].z;
        d[x]++; d[y]++; d[z]++;
    }
    m = amax(d, nv);
    EFREE(d);
    return m;
}

const float *mesh_read_get_vert(const MeshRead *q) { return q->rr; }
const int4  *mesh_read_get_tri(const MeshRead *q) { return q->tt; }
const int4  *mesh_read_get_dih(MeshRead *q) {
    if (!q->dd_computed) {
        UC(ini_dd(q));
        q->dd_computed = 1;
    }
    return q->dd;
}







// Grouped editable compilation unit for fragment indexing and half-edge I/O.



#include <vector_types.h>
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

#define FRAG_HOST
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
#undef FRAG_HOST

namespace frag_hst {

void estimates(int3 L, int nfrags, float maxd, /**/ int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = ncell(L, i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}

} // frag_hst





#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

enum {MAX_STRING_SIZE = 4096};
#ifdef __GNUC__
  #define __UNUSED __attribute__((unused))
#else
  #define __UNUSED
#endif
/* error codes */
enum {HE_OK, HE_IO, HE_MEMORY, HE_SYS, HE_USER, HE_INDEX, HE_NUM};

/* for user code */
#define ER(fmt, ...) ERR(HE_USER, (fmt), ##__VA_ARGS__)
#define ERR(code, fmt, ...)                                             \
    do {                                                                \
        he_err(code, __FILE__, __LINE__, (fmt), ##__VA_ARGS__);         \
        return code;                                                    \
    } while(0)

#define MSG(fmt, ...) he_msg(__FILE__, __LINE__, (fmt), ##__VA_ARGS__)

void he_err(int code, const char *file, int line, const char *fmt, ...);
void he_msg(const char *file, int line, const char *fmt, ...);

#define SIZE (MAX_STRING_SIZE)

void he_err(__UNUSED int code, const char *file, int line, const char *fmt, ...) {
    char msg[SIZE];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, SIZE, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%s:%d: %s\n", file, line, msg);
    abort();
}

void he_msg(const char *file, int line, const char *fmt, ...) {
    char msg[SIZE];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, SIZE, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%s:%d: %s\n", file, line, msg);
}



#include <stdio.h>
#define MALLOC(n, pp) he_malloc((n)*sizeof(**(pp)), (void**)(pp))
#define FREE(p) he_free(p)
#define MEMCPY(n, src, dest) he_memcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

int he_malloc(int, void**);
int he_free(void *ptr);
int he_memcpy(void *dest, const void *src, size_t n);

#define T HeHash

typedef struct T T;

int he_hash_ini(int n, T**);

/* hash[i, j] = v */
int he_hash_set(T*, int i, int j, int v);

/* ans = hash[i, j], ans = -1 if unset  */
int he_hash_get(T*, int i, int j);
int he_hash_fin(T*);

#undef T

#define T HeHash
#define E(fmt, ...) ERR(HE_INDEX, fmt, ##__VA_ARGS__);

typedef struct Node Node;
struct Node {
    int j;
    int v; /* value */
    Node  *nxt;
};
struct T {
    int n;
    Node **node;
};

int he_hash_ini(int n, T **pq) {
    T *q;
    int i;
    MALLOC(1, &q);
    MALLOC(q->n = n, &q->node);
    for (i = 0; i < n; i++)
        q->node[i] = NULL;
    *pq = q;
    return HE_OK;
}

static void free_node(Node *p) {
    Node *n;
    while (p != NULL) {
        n = p->nxt;
        FREE(p);
        p = n;
    }
}
int he_hash_fin(T *q) {
    int n, i;
    n = q->n;
    for (i = 0; i < n; i++)
        free_node(q->node[i]);
    FREE(q->node);
    FREE(q);
    return HE_OK;
}

int he_hash_set(T *q, int i, int j, int v) {
    int n;
    Node *nxt, *prv, *nnew;
    n = q->n;
    if (0 > i || i >= n || 0 > j || j >= n)
        E("[%d %d], n = %d", i, j, n);

    prv = q->node[i];
    while (prv != NULL) {
        if (prv->j == j) {
            prv->v = v;
            return HE_OK;
        }
        nxt = prv->nxt;
        if (nxt == NULL) break;
        prv = nxt;
    }

    MALLOC(1, &nnew);
    nnew->j = j;
    nnew->v = v;
    nnew->nxt = NULL;
    if (prv == NULL) q->node[i] = nnew;
    else prv->nxt = nnew;
    return HE_OK;
}

int he_hash_get(T *q, int i, int j) {
    int n;
    Node *node;
    n = q->n;
    if (0 > i || i >= n || 0 > j || j >= n)
        E("[%d %d], n = %d", i, j, n);
    node = q->node[i];
    while (node != NULL) {
        if (node->j == j) return node->v;
        node = node->nxt;
    }
    return -1;
}



#include <stdio.h>



#define T HeRead

struct int4;

typedef struct T T;

int he_read_ini(const char *path, T**);

/* tri = [t0, t1, t2] [t0, t2, t2], ... */
int he_read_tri_ini(int nv, int nt, int *tri, T**);

/* tri = [tri[0].x tri[0].y tri[0].z], ... [tri[0].x tri[0].y tri[0].z] */
int he_read_int4_ini(int nv, int nt, const int4 *tri, T**);

int he_read_fin(T*);

int he_read_nv(T*);
int he_read_nt(T*);
int he_read_ne(T*);
int he_read_nh(T*);

int he_read_nxt(T*, int**);
int he_read_flp(T*, int**);
int he_read_ver(T*, int**);
int he_read_tri(T*, int**);
int he_read_edg(T*, int**);

int he_read_hdg_ver(T*, int**);
int he_read_hdg_edg(T*, int**);
int he_read_hdg_tri(T*, int**);

int he_read_info(T*, FILE*);

#undef T

#define T He

typedef struct T T;
typedef struct HeRead HeRead;

int he_ini(HeRead*, T**);
int he_file_ini(const char *path, T**);
int he_fin(T*);

int he_nv(T*); /* number of vert, edg, edg, hdg */
int he_nt(T*);
int he_ne(T*);
int he_nh(T*);

int he_nxt(T*, int h); /* return half-edg */
int he_flp(T*, int h);

int he_ver(T*, int h); /* return ver, tri, or edg */
int he_tri(T*, int h);
int he_edg(T*, int h);

int he_hdg_ver(T*, int v); /* return half-edg */
int he_hdg_edg(T*, int e);
int he_hdg_tri(T*, int t);

int he_bnd(T*, int h); /* 1 for boundary */

#undef T

#define T He

#define MAGIC (43)

struct T {
    int nv, nt, ne, nh;
    int *nxt, *flp;
    int *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;
    int magic;
};

int he_file_ini(const char *path, T **pq) {
    HeRead *read;
    if (he_read_ini(path, &read) != HE_OK)
        ERR(HE_IO, "he_read_ini failed");
    he_ini(read, /**/ pq);
    he_read_fin(read);
    return HE_OK;
}

int he_ini(HeRead *r, T **pq) {
    T *q;
    int nv, nt, ne, nh;
    int *nxt, *flp, *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;
    MALLOC(1, &q);

    nv = q->nv = he_read_nv(r);
    nt = q->nt = he_read_nt(r);
    ne = q->ne = he_read_ne(r);
    nh = q->nh = he_read_nh(r);

    MALLOC(nh, &q->nxt); MALLOC(nh, &q->flp);
    MALLOC(nh, &q->ver); MALLOC(nh, &q->tri); MALLOC(nh, &q->edg);
    MALLOC(nv, &q->hdg_ver);
    MALLOC(ne, &q->hdg_edg);
    MALLOC(nt, &q->hdg_tri);

    he_read_nxt(r, &nxt);
    he_read_flp(r, &flp);
    he_read_ver(r, &ver);
    he_read_tri(r, &tri);
    he_read_edg(r, &edg);

    he_read_hdg_ver(r, &hdg_ver);
    he_read_hdg_edg(r, &hdg_edg);
    he_read_hdg_tri(r, &hdg_tri);

    MEMCPY(nh, nxt, q->nxt); MEMCPY(nh, flp, q->flp);
    MEMCPY(nh, ver, q->ver); MEMCPY(nh, tri, q->tri); MEMCPY(nh, edg, q->edg);
    MEMCPY(nv, hdg_ver, q->hdg_ver);
    MEMCPY(ne, hdg_edg, q->hdg_edg);
    MEMCPY(nt, hdg_tri, q->hdg_tri);

    q->magic = MAGIC;

    *pq = q;
    return HE_OK;
}

int he_fin(T *q) {
    if (q->magic != MAGIC)
        ERR(HE_MEMORY, "wrong fin() call");
    FREE(q->nxt); FREE(q->flp);
    FREE(q->ver); FREE(q->tri); FREE(q->edg);
    FREE(q->hdg_ver);
    FREE(q->hdg_edg);
    FREE(q->hdg_tri);
    FREE(q);
    return HE_OK;
}

int he_fin(T*);

int he_nv(T *q) { return q->nv; }
int he_nt(T *q) { return q->nt; }
int he_ne(T *q) { return q->ne; }
int he_nh(T *q) { return q->nh; }

/* validate */
#define V(i, n) if (0 > (i) || (i) >= n)                        \
        ERR(HE_INDEX, "%s=%d is not in [0, %d)", #i, i, n)
int he_nxt(T *q, int h) { V(h,q->nh); return q->nxt[h]; }
int he_flp(T *q, int h) {
    int f;
    V(h, q->nh);
    if ((f = q->flp[h]) == -1)
        ERR(HE_INDEX, "no flip for %d", h);
    return f;
}
int he_ver(T *q, int h) { V(h, q->nh); return q->ver[h]; }
int he_tri(T *q, int h) { V(h, q->nh); return q->tri[h]; }
int he_edg(T *q, int h) { V(h, q->nh); return q->edg[h]; }
int he_hdg_ver(T *q, int v) { V(v, q->nv); return q->hdg_ver[v]; }
int he_hdg_edg(T *q, int e) { V(e, q->ne); return q->hdg_edg[e]; }
int he_hdg_tri(T *q, int t) { V(t, q->nt); return q->hdg_tri[t]; }
int he_bnd(T *q, int h)     { V(h, q->nh); return q->flp[h] == -1; }



#include <stdio.h>
#include <stdlib.h>
#include <string.h>




int he_malloc(int size, /**/ void **pq) {
    void *q;
    q = malloc(size);
    if (q == NULL) ERR(HE_MEMORY, "malloc failed for size=%ld", size);
    *pq = q;
    return HE_OK;
}

int he_free(void *q) {
    free(q);
    q = NULL;
    return HE_OK;
}

int he_memcpy(void *dest, const void *src, size_t n) {
    if (dest == NULL) ERR(HE_MEMORY, "dest == NULL");
    if (src == NULL) ERR(HE_MEMORY, "src == NULL");
    memcpy(dest, src, n);
    return HE_OK;
}



#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vector_types.h>





/* string equal? */
int util_eq(const char *a, const char *b);

/* fgets without '\n' */
char *util_fgets(char *s, FILE *stream);

/* fgets without '\n' and skipping '#comments' */
char *util_comment_fgets(char *s, FILE *stream);




#define T HeRead
#define SIZE (MAX_STRING_SIZE)
#define E(fmt, ...) ERR(HE_IO, fmt, ##__VA_ARGS__);
#define MAGIC (42)

struct T {
    int nv, nt, ne, nh;
    int *nxt, *flp;
    int *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;
    int magic;
};


enum {HE_VALID_OK, HE_VALID_FAIL};
static int valid(int n, int *a, int *b) {
    int i, i0;
    for (i = 0; i < n; i++) {
        i0 = b[a[i]];
        if (i != i0) {
            MSG("i=%d a[i]=%d i0=%d ", i, a[i], i0);
            return HE_VALID_FAIL;
        }
    }
    return HE_VALID_OK;
}

static int alloc_hdg(int nh, T *q) {
    MALLOC(nh, &q->nxt); MALLOC(nh, &q->flp);
    MALLOC(nh, &q->ver); MALLOC(nh, &q->tri); MALLOC(nh, &q->edg);
    return HE_OK;
}
static int alloc_ver(int nv, T *q) { MALLOC(nv, &q->hdg_ver); return HE_OK; }
static int alloc_edg(int ne, T *q) { MALLOC(ne, &q->hdg_edg); return HE_OK; }
static int alloc_tri(int nt, T *q) { MALLOC(nt, &q->hdg_tri); return HE_OK; }
static int alloc(int nv, int ne, int nt, int nh, T *q) {
    alloc_hdg(nh, q);
    alloc_ver(nv, q); alloc_edg(ne, q); alloc_tri(nt, q);
    return HE_OK;
}

static int afree(T *q) {
    FREE(q->nxt); FREE(q->flp);
    FREE(q->ver); FREE(q->tri); FREE(q->edg);
    FREE(q->hdg_ver);
    FREE(q->hdg_edg);
    FREE(q->hdg_tri);
    return HE_OK;
}

static void setup_edg(T *q, HeHash *hdg, int i, int j,  int *pe) {
    int h, f, e;
    h = he_hash_get(hdg, i, j);
    if (h == -1) return;
    e = *pe;

    q->edg[h] = e;
    if ((f = he_hash_get(hdg, j, i)) != -1)
        q->edg[f] = e;

    he_hash_set(hdg, i, j, -1); /* counted it */
    he_hash_set(hdg, j, i, -1);

    e++;
    *pe = e;
}
static void setup_hdg(T *q, HeHash *hdg, int h, int t, int i, int j) {
    he_hash_set(hdg, i, j, h);
    q->hdg_tri[t] = q->hdg_ver[i] = h;
    q->ver[h] = i; q->tri[h] = t;
}
static void setup_flip(T *q, HeHash *hdg, int i, int j) {
    int h, f;
    h = he_hash_get(hdg, i, j);
    f = he_hash_get(hdg, j, i);
    q->flp[h] = f;
}

int he_read_int4_ini(int nv, int nt, const int4 *tt, T **pq) {
    int status;
    int *tri;
    int i, j;
    MALLOC(3*nt, &tri);
    for (j = i = 0; i < nt; i++) {
        tri[j++] = tt[i].x;
        tri[j++] = tt[i].y;
        tri[j++] = tt[i].z;
    }
    status = he_read_tri_ini(nv, nt, tri, pq);
    FREE(tri);
    return status;
}

int he_read_tri_ini(int nv, int nt, int *tri0, T **pq) {
    int ne, nh;
    int e, t, i, j, k;
    int h, hi, hj, hk;
    int *tri;
    T *q;
    HeHash *hdg; /* will maps [i, j] to half-edg */
    MALLOC(1, &q);

    alloc_ver(nv,        q);
    alloc_tri(nt,        q);
    alloc_hdg(nh = 3*nt, q);
    he_hash_ini(nv, &hdg);
    for (tri = tri0, t = h = 0; t < nt; t++) {
        i = *tri++; j = *tri++; k = *tri++;
        assert(i < nv); assert(j < nv); assert(k < nv);
        setup_hdg(q, hdg, hi = h++, t, i, j);
        setup_hdg(q, hdg, hj = h++, t, j, k);
        setup_hdg(q, hdg, hk = h++, t, k, i);
        q->nxt[hi] = hj; q->nxt[hj] = hk; q->nxt[hk] = hi;
    }

    for (tri = tri0, t = h = 0; t < nt; t++) {
        i = *tri++; j = *tri++; k = *tri++;
        setup_flip(q, hdg, i, j);
        setup_flip(q, hdg, j, k);
        setup_flip(q, hdg, k, i);
    }

    for (tri = tri0, ne = t = h = 0; t < nt; t++) {
        i = *tri++; j = *tri++; k = *tri++;
        setup_edg(q, hdg, i, j,  &ne);
        setup_edg(q, hdg, j, k,  &ne);
        setup_edg(q, hdg, k, i,  &ne);
    }
    he_hash_fin(hdg);

    alloc_edg(ne, q);
    for (h = 0; h < nh; h++) {
        e = q->edg[h];
        q->hdg_edg[e] = h;
    }

    q->nv = nv; q->ne = ne; q->nt = nt; q->nh = nh;
    q->magic = MAGIC;
    *pq = q;
    return HE_OK;
}

int he_read_ini(const char *path, T **pq) {
    T *q;
    FILE *f;
    char line[SIZE];
    int i, cnt;
    int nv, nt, ne, nh;

#   define NXT() if (util_fgets(line, f) == NULL)  \
        E("unexpected EOF in '%s'", path)
    MALLOC(1, &q);
    f = fopen(path, "r");
    if (f == NULL) E("fail to open '%s'", path);
    NXT();
    if (!util_eq(line, "HE"))
        E("'%s' is not a he file", path);
    NXT();
    cnt = sscanf(line, "%d %d %d %d", &nv, &ne, &nt, &nh);
    if (cnt != 4)
        E("'%s' != [nv nt ne nh] in '%s'", line, path);
    if (nv <= 0 || nt <= 0 || ne <= 0 || nh <= 0)
        E("wrong sizes '%s' in '%s'", line, path);

    alloc(nv, ne, nt, nh, /**/ q);

    for (i = 0; i < nh; i++) {
        NXT();
        cnt = sscanf(line, "%d %d  %d %d %d",
                     &q->nxt[i], &q->flp[i], &q->ver[i], &q->edg[i], &q->tri[i]);
        if (cnt != 5) E("wrong half-edg line '%s' in '%s'", line, path);
    }
    for (i = 0; i < nv; i++) {
        NXT();
        cnt = sscanf(line, "%d", &q->hdg_ver[i]);
        if (cnt != 1) E("wrong ver line '%s' in '%s'", line, path);
    }
    for (i = 0; i < ne; i++) {
        NXT();
        cnt = sscanf(line, "%d", &q->hdg_edg[i]);
        if (cnt != 1) E("wrong edg line '%s' in '%s'", line, path);
    }
    for (i = 0; i < nt; i++) {
        NXT();
        cnt = sscanf(line, "%d", &q->hdg_tri[i]);
        if (cnt != 1) E("wrong tri line '%s' in '%s'", line, path);
    }
    if (fclose(f) != 0)
        E("fail to close file '%s'", path);

    q->nv = nv; q->nt = nt; q->ne = ne; q->nh = nh;
    q->magic = MAGIC;

    if (valid(nv, q->hdg_ver, q->ver) != HE_VALID_OK)
        E("invalid ver references");
    if (valid(ne, q->hdg_edg, q->edg) != HE_VALID_OK)
        E("invalid edg references");
    if (valid(nt, q->hdg_tri, q->tri) != HE_VALID_OK)
        E("invalid tri references");

    *pq = q;
    return HE_OK;
}

int he_read_fin(T *q) {
    if (q->magic != MAGIC)
        ERR(HE_MEMORY, "wrong fin() call");
    afree(q);
    FREE(q);
    return HE_OK;
}

int he_read_info(T *q, FILE *f) {
    int r;
    int nv, nt, ne, nh;
    int *nxt, *flp, *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;

    nv = q->nv; nt = q->nt; ne = q->ne; nh = q->nh;
    nxt = q->nxt; flp = q->flp;
    ver = q->ver; tri = q->tri; edg = q->edg;
    hdg_ver = q->hdg_ver;
    hdg_edg = q->hdg_edg;
    hdg_tri = q->hdg_tri;

    r = fprintf(f, "%d %d %d %d\n", nv, ne, nt, nh);
    if (r <= 0)
        ERR(HE_IO, "fprintf() failed");

    fprintf(f, "[nh=%d lines]\n", nh);
    fprintf(f, "%d %d %d %d %d\n",
            nxt[0], flp[0], ver[0], tri[0], edg[0]);
    fputs("...\n", f);
    fprintf(f, "%d %d %d %d %d\n",
            nxt[nh-1], flp[nh-1], ver[nh-1], tri[nh-1], edg[nh-1]);
    fprintf(f, "[nv=%d lines]\n", nv);
    fprintf(f, "%d\n", hdg_ver[0]);
    fputs("...\n", f);
    fprintf(f, "%d\n", hdg_ver[nv-1]);

    fprintf(f, "[ne=%d lines]\n", ne);
    fprintf(f, "%d\n", hdg_edg[0]);
    fputs("...\n", f);
    fprintf(f, "%d\n", hdg_edg[ne-1]);

    fprintf(f, "[nt=%d lines]\n", nt);
    fprintf(f, "%d\n", hdg_tri[0]);
    fputs("...\n", f);
    fprintf(f, "%d\n", hdg_tri[nt-1]);

    return HE_OK;
}

int he_read_nv(T *q) { return q->nv; }
int he_read_nt(T *q) { return q->nt; }
int he_read_ne(T *q) { return q->ne; }
int he_read_nh(T *q) { return q->nh; }

int he_read_nxt(T *q, int **p) { *p = q->nxt; return HE_OK; };
int he_read_flp(T *q, int **p) { *p = q->flp; return HE_OK; };
int he_read_ver(T *q,  int **p) { *p = q->ver; return HE_OK; };
int he_read_tri(T *q,  int **p) { *p = q->tri; return HE_OK; };
int he_read_edg(T *q,  int **p) { *p = q->edg; return HE_OK; };

int he_read_hdg_ver(T *q, int **p) { *p = q->hdg_ver; return HE_OK; };
int he_read_hdg_edg(T *q, int **p) { *p = q->hdg_edg; return HE_OK; };
int he_read_hdg_tri(T *q, int **p) { *p = q->hdg_tri; return HE_OK; };



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>






#define  SIZE (MAX_STRING_SIZE)

int util_eq(const char *a, const char *b) {
    return strncmp(a, b, SIZE) == 0;
}

char *util_fgets(char *s, FILE *stream) {
    char *c;
    if (fgets(s, SIZE, stream) == NULL)
        return NULL;
    if ((c = strchr(s, '\n')) != NULL)
        *c = '\0';
    return s;
}

enum {NO, YES};
static int commentp(const char *s) {
    while (*(s++) != '\0') {
        if    (*s == '#') return YES;
        if (!isblank(*s)) return NO;
    }
    return YES;
}
char *util_comment_fgets(char *s, FILE *stream) {
    do if (util_fgets(s, stream) == NULL)
           return NULL;
    while (commentp(s) == YES);
    return s;
}

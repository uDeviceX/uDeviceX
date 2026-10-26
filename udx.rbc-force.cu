// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/force/conf.cpp */
#include <stdio.h>
#include <string.h>

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
/* begin src/io/mesh_read/imp.h */
struct int4;
struct MeshRead;

// tag::interface[]
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
// end::interface[]
/* end src/io/mesh_read/imp.h */

/* begin src/rbc/force/imp.h */
struct RbcParams;
struct RbcForce;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void rbc_force_ini(const MeshRead*, RbcForce**);
void rbc_force_fin(RbcForce*);
// end::mem[]

// tag::set[]
void rbc_force_set_stressful(int nt, float totArea, /**/ RbcForce*); // <1>
void rbc_force_set_stressfree(const char *fname, /**/ RbcForce*);    // <2>

void rbc_force_set_rnd0(RbcForce*);           // <3>
void rbc_force_set_rnd1(int seed, RbcForce*); // <4>
void rbc_force_set_bending(const MeshRead*, const char*, RbcForce*);
// end::set[]

// tag::cnf[]
void rbc_force_set_conf(const MeshRead*, const Config *cfg, const char *name, RbcForce*);
// end::cnf[]

// tag::apply[]
void rbc_force_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*, /**/ Force*); // <1>
void rbc_force_stat(/**/ float *pArea, float *pVolume); // <2>
// end::apply[]
/* end src/rbc/force/imp.h */

void rbc_force_set_conf(const MeshRead *cell, const Config *cfg, const char *name, RbcForce *q) {
    int stress_free, rnd;
    const char *bending;
    UC(conf_lookup_bool_ns(cfg, name, "stress_free", &stress_free));
    UC(conf_lookup_bool_ns(cfg, name, "rnd", &rnd));
    UC(conf_lookup_string_ns(cfg, name, "bending", &bending));
    UC(rbc_force_set_bending(cell, bending, /**/ q));

    if (stress_free) {
        const char *fname;
        UC(conf_lookup_string_ns(cfg, name, "stress_free_file", &fname));
        UC(rbc_force_set_stressfree(fname, /**/ q));
    }
    else {
        float Atot = 0;
        int nt;
        nt = mesh_read_get_nt(cell);
        UC(conf_lookup_float_ns(cfg, name, "totArea", &Atot));
        UC(rbc_force_set_stressful(nt, Atot, /**/ q));
    }

    if (rnd) {
        int seed;
        UC(conf_lookup_int_ns(cfg, name, "seed", &seed));
        UC(rbc_force_set_rnd1(seed, q));
    }
    else {
        UC(rbc_force_set_rnd0(q));
    }
}
/* end src/rbc/force/conf.cpp */

#line 1 "src/rbc/force/imp.cu"


/* begin src/rbc/force/imp.cu */
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


#ifndef UDX_RBC_FORCE_DEF_COMMON
#define UDX_RBC_FORCE_DEF_COMMON
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
/* begin src/inc/type.h */
#ifndef UDX_RBC_FORCE_TYPE_COMMON
#define UDX_RBC_FORCE_TYPE_COMMON
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
/* begin src/math/dev.h */
#ifndef UDX_RBC_FORCE_MATH_COMMON
#define UDX_RBC_FORCE_MATH_COMMON
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
#endif
/* begin src/math/tri/dev.h */
#ifdef TRI_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace tri_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace tri_dev {
  #define END }
#endif

BEGIN

_S_ void diff(const double *a, const double *b, /**/ double  *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
_S_ double vabs(double *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}
_S_ void swap(double *a, double *b) {
    double t;
    t = *a; *a = *b; *b = t;
}
_S_ int less(double *a, double *b) { return (*a) < (*b); }
_S_ void  sort3(double *a, double *b, double *c) {
    if (less(c, b)) swap(c, b);
    if (less(b, a)) swap(b, a);
    if (less(c, b)) swap(c, b);
}

_I_ double kahan_area0(double a, double b, double c) {
    sort3(&c, &b, &a); /* make a > b > c */
    return sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4;
}

_I_ double kahan_area(const double r0[3], const double r1[3], const double r2[3]) {
    double r01[3], r12[3], r20[3], a, b, c;
    diff(r0, r1, /**/ r01);
    diff(r1, r2, /**/ r12);
    diff(r2, r0, /**/ r20);
    a = vabs(r01); b = vabs(r12); c = vabs(r20);
    return kahan_area0(a, b, c);
}

_S_ double orient2d_xy(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = X, Q = Y};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_zx(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Z, Q = X};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_yz(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Y, Q = Z};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_I_ double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]) {
    enum {X, Y, Z};
    double ax, ay, az, bx, by, bz, cx, cy, cz;

    ax = a[X] - d[X]; ay = a[Y] - d[Y]; az = a[Z] - d[Z];
    bx = b[X] - d[X]; by = b[Y] - d[Y]; bz = b[Z] - d[Z];
    cx = c[X] - d[X]; cy = c[Y] - d[Y]; cz = c[Z] - d[Z];

    return ax*(by*cz - bz*cy) + bx*(cy*az - cz*ay) + cx*(ay*bz - az*by);
}

_I_ void ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]) {
    enum {X, Y, Z};
    r[X] = orient2d_yz(a, b, c);
    r[Y] = orient2d_zx(a, b, c);
    r[Z] = orient2d_xy(a, b, c);
}

_I_ double shewchuk_area(const double a[3], const double b[3], const double c[3]) {
    double x, y, z;
    x = orient2d_yz(a, b, c);
    y = orient2d_zx(a, b, c);
    z = orient2d_xy(a, b, c);
    return sqrt(x*x + y*y + z*z)/2;
}

_I_ void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                       double *px, double *py) {
    enum {X, Y, Z};
    double bcx, bcy, bcz, bc;
    double k[3], l[3];

    ac_bc_cross(b, c, a, /**/ k);
    ac_bc_cross(c, b, d, /**/ l);

    bcx = b[X] - c[X];
    bcy = b[Y] - c[Y];
    bcz = b[Z] - c[Z];
    bc = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);

    *py = -orient3d(a, b, c, d)*bc;
    *px = k[X]*l[X] + k[Y]*l[Y] + k[Z]*l[Z];
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
/* end src/math/tri/dev.h */



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
#ifndef UDX_RBC_FORCE_DEVICE_API_COMMON
#define UDX_RBC_FORCE_DEVICE_API_COMMON
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
#ifndef UDX_RBC_FORCE_KL_COMMON
#define UDX_RBC_FORCE_KL_COMMON
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

/* begin src/rbc/params/imp.h */
struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
    float totArea, totVolume;
};

// tag::mem[]
void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);
// end::mem[]

// tag::set[]
void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);
void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *);
// end::set[]

// tag::get[]
double rbc_params_get_tot_volume(const RbcParams *);
// end::get[]

// tag::cnf[]
void rbc_params_set_conf(const Config *c, const char *name, RbcParams *);
// end::cnf[]

// tag::view[]
RbcParams_v rbc_params_get_view(const RbcParams *);
// end::view[]
/* end src/rbc/params/imp.h */
/* begin src/rbc/adj/imp.h */
struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

// tag::interface[]
void adj_ini(int md, int nt, int nv, const int4 *tt, /**/ Adj**);
void adj_fin(Adj*);
int  adj_get_map(int i, const Adj*, /**/ AdjMap *m);
int  adj_get_max(const Adj*);
void adj_get_anti(const Adj*, /**/ int *anti);

void adj_view_ini(const Adj*, Adj_v**);
void adj_view_fin(Adj_v*);
// end::interface[]
/* end src/rbc/adj/imp.h */
/* begin src/rbc/shape/imp.h */
struct RbcShape;
struct MeshRead;
struct Adj;

/* rr: vertices: x0 y0 z0   x1 y1 z1 ... */

// tag::mem[]
void rbc_shape_ini(const Adj*, const float *rr, /**/ RbcShape**);
void rbc_shape_fin(/**/ RbcShape*);
// end::mem[]

// tag::int[]
void rbc_shape_edg (RbcShape*, /**/ float**); // <1>
void rbc_shape_area(RbcShape*, /**/ float**); // <2>

void rbc_shape_total_area(RbcShape*, /**/ float*);    // <3>
void rbc_shape_total_volume(RbcShape*, /**/ float*);  // <4>
// end::int[]
/* end src/rbc/shape/imp.h */
/* begin src/rbc/force/bending/imp.h */
struct RbcParams;
struct Bending;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void bending_kantor_ini(const MeshRead*, Bending**);
void bending_juelicher_ini(const MeshRead*, Bending**);
void bending_none_ini(const MeshRead*, Bending**);

void bending_fin(Bending*);
// end::mem[]

// tag::apply[]
void bending_apply(Bending*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]
/* end src/rbc/force/bending/imp.h */

/* begin src/rbc/force/rnd/imp.h */
struct RbcRnd;

// tag::enum[]
#ifndef UDX_RBC_FORCE_SEED_TIME
#define UDX_RBC_FORCE_SEED_TIME
enum {SEED_TIME = -1}; /* magic seed: ini from "time" */
#endif
// end::enum[]

// tag::mem[]
void rbc_rnd_ini(int n, long seed, RbcRnd**);
void rbc_rnd_fin(RbcRnd*);
// end::mem[]

// tag::int[]
void rbc_rnd_gen(RbcRnd*, int n, /**/ float**); // <1>
float rbc_rnd_get_hst(const RbcRnd*, int i);    // <2>
// end::int[]
/* end src/rbc/force/rnd/imp.h */
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

/* begin src/rbc/adj/type/common.h */
// tag::interface[]
struct AdjMap {
    int i0, i1, i2, i3, i4;
    int rbc;
};
// end::interface[]
/* end src/rbc/adj/type/common.h */
/* begin src/rbc/adj/type/dev.h */
struct Adj_v {
    int *adj0;
    int *adj1;
    int nv, md;
};

/* end src/rbc/adj/type/dev.h */
/* begin src/rbc/adj/dev.h */
static __device__ int fetch(const int *a, int i) { return a[i]; }
static __device__ int adj_get_map0(int md, int nv, int i, const int *adj0, const int *adj1, /**/ AdjMap *m) {
    int i0, i1, i2, i3, i4;
    int rbc, offset, j, k;
    i0 = i / md;
    j  = i % md;

    k   = (i0 % nv) * md;
    i1 = fetch(adj0, k + j);
    if (i1 == -1) return 0; /* invalid */

    i2 = fetch(adj0, k + ((j + 1) % md));
    if (i2 == -1) {
        i2 = fetch(adj0, k    );
        i3 = fetch(adj0, k + 1);
    } else {
        i3 = fetch(adj0, k + ((j + 2) % md));
        if (i3 == -1) i3 = fetch(adj0, k);
    }
    i4 = fetch(adj1, k + j);

    rbc = i0 / nv;
    offset = rbc * nv;
    i1 += offset; i2 += offset; i3 += offset; i4 += offset; /* no i0 */

    assert(rbc < MAX_CELL_NUM);

    m->rbc = rbc;
    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    return 1;
}

// tag::interface[]
static __device__ int adj_get_map(int i, Adj_v *adj, /**/ AdjMap *m) {
// end::interface[]
    int md, nv;
    md = adj->md;
    nv = adj->nv;
    const int *adj0 = adj->adj0;
    const int *adj1 = adj->adj1;
    return adj_get_map0(md, nv, i, adj0, adj1, m);
}
/* end src/rbc/adj/dev.h */
/* begin src/rbc/force/area_volume/imp.h */
struct AreaVolume;
struct Particle;
struct int4;

// tag::interface[]
void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**); // <1>
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*, /**/ float **av); // <2>
void area_volume_host(AreaVolume*, /**/ float **av); // <3>
// end::interface[]
/* end src/rbc/force/area_volume/imp.h */


/* begin src/rbc/force/imp/type.h */
enum {
    RBC_SFUL,
    RBC_SFREE
};

struct StressFree_v {
    float *ll; /* eq. spring lengths */
    float *aa; /* eq. triangle areas */
};

struct StressFul_v {
    float l0; /* eq. spring length */
    float a0; /* eq. triangle area */
};

union SFreeInfo {
    StressFul_v sful;
    StressFree_v sfree;
};

enum {
    RBC_RND0,
    RBC_RND1,
};

struct Rnd0_v {};
struct Rnd1_v {
    int *anti; /* indices of anti edges */
    float *rr; /* random numbers (managed by RbcRnd) */
};

union RndInfo {
    Rnd0_v rnd0;
    Rnd1_v rnd1;
};

struct RbcForce {
    RbcRnd *rnd;
    Bending *bending;

    Adj   *adj;
    Adj_v *adj_v;

    int stype;
    SFreeInfo sinfo;
    int rtype;
    RndInfo rinfo;
};
/* end src/rbc/force/imp/type.h */

namespace rbc_force_dev {
/* begin src/rbc/force/dev/type.h */
struct StressInfo { double l0, a0; };
struct Rnd0Info {};
struct Rnd1Info { double r; };
/* end src/rbc/force/dev/type.h */
/* begin src/rbc/force/dev/fetch.h */
static __device__ StressInfo fetch_stress_info(int, StressFul_v sv) {
    StressInfo si;
    si.l0 = sv.l0;
    si.a0 = sv.a0;
    return si;
}

static __device__ StressInfo fetch_stress_info(int i, StressFree_v sv) {
    StressInfo si;
    si.l0 = sv.ll[i];
    si.a0 = sv.aa[i];
    return si;
}

static __device__ Rnd0Info fetch_rnd_info(int, int, Rnd0_v rv) {
    Rnd0Info ri;
    return ri;
}

static __device__ Rnd1Info fetch_rnd_info(int i0, int j, Rnd1_v rv) {
    /* i0: edge index; j: vertex index */
    Rnd1Info ri;
    int i1;
    i1 = rv.anti[i0];
    if (i1 > i0) j = j - i0 + i1;
    ri.r = rv.rr[j];
    return ri;
}
/* end src/rbc/force/dev/fetch.h */
/* begin src/rbc/force/dev/common.h */
static __device__ double3 fvolume(const RbcParams_v *par, double3 r2, double3 r3, double v0, double v) {
    double f0;
    double3 f;
    cross(&r3, &r2, /**/ &f);
    f0 = par->kv * (v - v0) / (6 * v0);
    scal(f0, /*io*/ &f);
    return f;
}

static __device__ double3 farea(const RbcParams_v *par, double3 x21, double3 x31, double3 x32,   double a0, double A0, double A) {
    double3 nn, f;
    double a, f0, fa, fA;
    cross(&x21, &x31, /**/ &nn);
    cross(&nn, &x32,  /**/ &f);
    a = 0.5 * sqrt(dot<double>(&nn, &nn));

    fA = - par->ka * (A - A0) / (4 * A0 * a);
    fa = - par->kd * (a - a0) / (4 * a0 * a);
    f0 = fA + fa;
    scal(f0, /*io*/ &f);
    return f;
}

enum {SPRING_OK, SPRING_LONG};
static __device__ double sq(double x) { return x * x; }
static __device__ double wlc0(double r) { return (4*sq(r)-9*r+6)/(4*sq(r-1)); }
static __device__ double wlc(double lmax, double ks, double r) { return ks/lmax*wlc0(r/lmax); }
static __device__ double3 fspring(const RbcParams_v *par, double3 x21, double l0, /**/ int *pstatus) {
  #define wlc_r(r) (wlc(lmax, ks, r))
    double m, r, fwlc, fpow, lmax, ks, x0;
    *pstatus = SPRING_OK;
    ks = par->ks; m = par->mpow; x0 = par->x0;
    r = sqrtf(dot<double>(&x21, &x21));
    lmax = l0 / x0;
    if (r >= lmax) {
        *pstatus = SPRING_LONG;
        r = RBC_SPRING_CAP * lmax;
    }
    fwlc =   wlc_r(r); /* make fwlc + fpow = 0 for r = l0 */
    fpow = - wlc_r(l0) * powf(l0, m + 1) / powf(r, m + 1);
    scal(fwlc + fpow, /*io*/ &x21);
    return x21;
  #undef wlc_r
}

static __device__ void report_tri(double3 r1, double3 r2, double3 r3) {
    printf("bad triangle: [%g %g %g] [%g %g %g] [%g %g %g]\n",
           r1.x, r1.y, r1.z,   r2.x, r2.y, r2.z,   r3.x, r3.y, r3.z);
    assert(0);
}
static __device__ double3 ftri(const RbcParams_v *par, double3 r1, double3 r2, double3 r3,
                             StressInfo si, double area, double volume) {
    int spring_status;
    double3 f, fv, fs;
    double3 x21, x32, x31;

    diff(&r2, &r1, /**/ &x21);
    diff(&r3, &r2, /**/ &x32);
    diff(&r3, &r1, /**/ &x31);

    f = farea(par, x21, x31, x32, si.a0, par->totArea, area);
    fv = fvolume(par, r2, r3, par->totVolume, volume);
    add(&fv, /*io*/ &f);

    fs = fspring(par, x21, si.l0, &spring_status);
#ifdef RBC_SPRING_FAIL
    if (spring_status != SPRING_OK) report_tri(r1, r2, r3);
#endif
    add(&fs, /*io*/ &f);
    return f;
}

static __device__ double3 fvisc(const RbcParams_v *par, double3 r1, double3 r2, double3 u1, double3 u2) {
    const double gammaC = par->gammaC, gammaT = par->gammaT;
    double3 du, dr, f = make_double3(0, 0, 0);
    diff(&u2, &u1, /**/ &du);
    diff(&r1, &r2, /**/ &dr);

    double fac = dot<double>(&du, &dr) / dot<double>(&dr, &dr);

    axpy(gammaT      , &du, /**/ &f);
    axpy(gammaC * fac, &dr, /**/ &f);

    return f;
}

static __device__ double3 frnd(double, const RbcParams_v*, double3, double3, Rnd0Info) {
    double3 f;
    f.x = f.y = f.z = 0;
    return f;
}

static __device__ double  frnd0(double dt, const RbcParams_v *par, double rnd) {
    double f, g, T;
    g = par->gammaC; T = par->kBT;
    f  = sqrtf(2*g*T/dt)*rnd;
    return f;
}

static __device__ double3 frnd(double dt, const RbcParams_v *par, double3 r1, double3 r2, Rnd1Info rnd) {
    double3 dr;
    double r, f0;
    diff(&r1, &r2, /**/ &dr);
    r = sqrtf(dot<double>(&dr, &dr));
    f0 = frnd0(dt, par, rnd.r);
    scal(f0/r, /*io*/ &dr);
    return dr;
}
/* end src/rbc/force/dev/common.h */
/* begin src/rbc/force/dev/main.h */
struct Part { double3 r, v; };
typedef double3 Pos;

static __device__ Pos fetchPos(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pos r;
    r.x = pp[i].r[X];
    r.y = pp[i].r[Y];
    r.z = pp[i].r[Z];
    return r;
}

static __device__ Part fetchPart(const Particle *pp, int i) {
    enum {X, Y, Z};
    Part p;
    const float *r, *v;
    r = pp[i].r; v = pp[i].v;
    p.r.x = r[X]; p.r.y = r[Y]; p.r.z = r[Z];
    p.v.x = v[X]; p.v.y = v[Y]; p.v.z = v[Z];
    return p;
}

template <typename RndInfo>
static __device__ void adj_tris(double dt,
                                const RbcParams_v *par, const Particle *pp,  const Part p0, const float *av,
                                const StressInfo si, RndInfo ri,
                                AdjMap *m, /*io*/ double f[3]) {
    enum {X, Y, Z};
    double3 f0;
    int i1, i2, rbc;
    double area, volume;
    i1 = m->i1; i2 = m->i2; rbc = m->rbc;

    const Part p1 = fetchPart(pp, i1);
    const Pos  r2 = fetchPos(pp,  i2);

    area = av[2*rbc]; volume = av[2 * rbc + 1];
    f0 = ftri(par, p0.r, p1.r, r2, si, area, volume);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;

    f0 = fvisc(par, p0.r, p1.r, p0.v, p1.v);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;

    f0 = frnd(dt, par, p0.r, p1.r, ri);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;
}

template <typename Stress_v, typename Rnd_v>
__global__ void force(double dt,
                      RbcParams_v par, int md, int nv, int nc, const Particle *pp,
                      Adj_v adj,
                      Stress_v sv, Rnd_v rv,
                      const float *av, /**/ float *ff) {
    enum {X, Y, Z};
    int i, pid, valid;
    double f[3];
    AdjMap m;
    StressInfo si;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;

    if (pid >= nc * nv) return;
    valid = adj_get_map(i, &adj, /**/ &m);
    if (!valid) return;
    si = fetch_stress_info(i % (md * nv), sv);
    auto ri = fetch_rnd_info(i % (md * nv), i, rv);

    const Part p0 = fetchPart(pp, m.i0);

    f[X] = f[Y] = f[Z] = 0;
    adj_tris(dt, &par, pp, p0, av, si, ri, &m, /*io*/ f);

    atomicAdd(&ff[3 * pid + 0], f[X]);
    atomicAdd(&ff[3 * pid + 1], f[Y]);
    atomicAdd(&ff[3 * pid + 2], f[Z]);
}
/* end src/rbc/force/dev/main.h */
}

/* begin src/rbc/force/imp/main.h */
static bool is_stress_free(const RbcForce *f) {
    return f->stype == RBC_SFREE;
}

static bool is_rnd(const RbcForce *f) {
    return f->rtype == RBC_RND1;
}

void rbc_force_ini(const MeshRead *cell, RbcForce **pq) {
    RbcForce *q;
    int md, nt, nv;
    const int4 *tt;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    md = mesh_read_get_md(cell);
    tt = mesh_read_get_tri(cell);

    UC(adj_ini(md, nt, nv, tt, /**/ &q->adj));
    UC(adj_view_ini(q->adj, /**/ &q->adj_v));

    *pq = q;
}

static void fin_rnd(RbcRnd *rnd) {
    rbc_rnd_fin(rnd);
}

static void fin_stress(RbcForce *f) {
    if (is_stress_free(f)) {
        StressFree_v v = f->sinfo.sfree;
        Dfree(v.ll);
        Dfree(v.aa);
    }
}

void rbc_force_fin(RbcForce *q) {
    if (is_rnd(q)) {
        Rnd1_v v = q->rinfo.rnd1;
        Dfree(v.anti);
        UC(fin_rnd(q->rnd));
    }
    UC(fin_stress(q));
    UC(adj_fin(q->adj));
    UC(adj_view_fin(q->adj_v));
    UC(bending_fin(q->bending));
    EFREE(q);
}

void rbc_force_set_stressful(int nt, float totArea, /**/ RbcForce *f) {
    StressFul_v v;
    float a0 = totArea / nt;

    v.a0 = a0;
    v.l0 = sqrt(a0 * 4.0 / sqrt(3.0));

    f->stype = RBC_SFUL;
    f->sinfo.sful = v;
}

void rbc_force_set_stressfree(const char *fname, /**/ RbcForce *f) {
    StressFree_v v;
    MeshRead *cell;
    const Adj *adj;
    RbcShape *shape;
    const float *rr;
    float *ll_hst, *aa_hst;
    int n;

    adj = f->adj;
    UC(mesh_read_ini_off(fname, &cell));
    rr = mesh_read_get_vert(cell);

    UC(rbc_shape_ini(adj, rr, /**/ &shape));

    n = adj_get_max(adj);

    rbc_shape_edg(shape, &ll_hst);
    rbc_shape_area(shape, &aa_hst);

    Dalloc(&v.ll, n);
    Dalloc(&v.aa, n);

    cH2D(v.ll, ll_hst, n);
    cH2D(v.aa, aa_hst, n);

    UC(rbc_shape_fin(shape));
    UC(mesh_read_fin(cell));

    f->stype = RBC_SFREE;
    f->sinfo.sfree = v;
}

void rbc_force_set_rnd0(RbcForce *f) {
    Rnd0_v rnd0;
    f->rtype = RBC_RND0;
    f->rinfo.rnd0 = rnd0;
}

void rbc_force_set_rnd1(int seed, RbcForce *f) {
    Rnd1_v rnd1;
    int n;
    int *hst;
    Adj *adj = f->adj;

    n = adj_get_max(adj);
    Dalloc(&rnd1.anti, n);

    EMALLOC(n, &hst);
    adj_get_anti(adj, /**/ hst);
    cH2D(rnd1.anti, hst, n);
    EFREE(hst);

    UC(rbc_rnd_ini(n * MAX_CELL_NUM, seed, &f->rnd));

    f->rtype = RBC_RND1;
    f->rinfo.rnd1 = rnd1;
}

void rbc_force_set_bending(const MeshRead *cell, const char *type, RbcForce *q) {
    if (same_str(type, "kantor"))
        UC(bending_kantor_ini(cell, /**/ &q->bending));
    else if (same_str(type, "juelicher"))
        UC(bending_juelicher_ini(cell, /**/ &q->bending));
    else if (same_str(type, "none"))
        UC(bending_none_ini(cell, /**/ &q->bending));
    else
        ERR("unknown bending type '%s'", type);
    msg_print("bending: %s", type);
}
/* end src/rbc/force/imp/main.h */
/* begin src/rbc/force/imp/forces.h */
static void get_stress_view(const RbcForce *f, /**/ StressFul_v *v) {
    *v = f->sinfo.sful;
}

static void get_stress_view(const RbcForce *f, /**/ StressFree_v *v) {
    *v = f->sinfo.sfree;
}

static void get_rnd_view(const RbcForce *f, /**/ Rnd0_v *v) {
    *v = f->rinfo.rnd0;
}

static void get_rnd_view(const RbcForce *f, /**/ Rnd1_v *v) {
    *v = f->rinfo.rnd1;
}

template <typename Stress_v, typename Rnd_v>
static void apply(float dt, RbcParams_v parv, int nc, int nv, int md,
                  const Particle *pp,
                  const Adj_v *adj_v, Stress_v sv, Rnd_v rv,
                  float *av, /**/ Force *ff) {
    if (!d::is_device_pointer(ff))  ERR("`ff` is not a device pointer");
    KL(rbc_force_dev::force, (k_cnf(nc*nv*md)), (dt, parv, md, nv, nc, pp,
                                                 *adj_v, sv, rv, av, /**/ (float*)ff));
}

template <typename Stress_v>
static void dispatch_rnd(float dt, RbcParams_v parv, int nc, int nv, int md,
                         const Particle *pp, RbcRnd *rnd,
                         const Adj_v *adj_v, RbcForce *t, Stress_v sv,
                         float *av, /**/ Force *ff){
    if (is_rnd(t)) {
        Rnd1_v rv;
        get_rnd_view(t, &rv);
        rbc_rnd_gen(rnd, nc * md * nv, /**/ &rv.rr);
        apply(dt, parv, nc, nv, md, pp, adj_v, sv, rv, av, /**/ ff);
    }
    else {
        Rnd0_v rv;
        get_rnd_view(t, &rv);
        apply(dt, parv, nc, nv, md, pp, adj_v, sv, rv, av, /**/ ff);
    }
}

void rbc_force_apply(RbcForce *t, const RbcParams *par, float dt, const RbcQuants *q, /**/ Force *ff) {
    RbcParams_v parv;
    float *av;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))
        ERR("`q->pp` is not a device pointer");

    parv = rbc_params_get_view(par);
    UC(area_volume_compute(q->area_volume, q->nc, q->pp, /**/ &av));
    UC(bending_apply(t->bending, par, q, /*io*/ ff));

    if (is_stress_free(t)) {
        StressFree_v si;
        get_stress_view(t, &si);
        dispatch_rnd(dt, parv, q->nc, q->nv, q->md, q->pp, t->rnd,
                     t->adj_v, t, si, av, /**/ ff);
    }
    else {
        StressFul_v si;
        get_stress_view(t, &si);
        dispatch_rnd(dt, parv, q->nc, q->nv, q->md, q->pp, t->rnd,
                     t->adj_v, t, si, av, /**/ ff);
    }
}
/* end src/rbc/force/imp/forces.h */
/* begin src/rbc/force/imp/stat.h */
void rbc_force_stat(float *pArea, float *pVolume) {
    *pArea = 1.0; *pVolume = 1.0; /* TODO */
}
/* end src/rbc/force/imp/stat.h */
/* end src/rbc/force/imp.cu */

#line 1 "src/rbc/force/rnd/api/imp.cpp"

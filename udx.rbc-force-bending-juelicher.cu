// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/force/bending/juelicher/imp.cu */
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
/* begin src/he/read.h */
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
/* end src/he/read.h */
/* begin src/he/he.h */
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
/* end src/he/he.h */

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

/* begin src/rbc/force/bending/juelicher/imp.h */
struct RbcParams;
struct Juelicher;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void juelicher_ini(const MeshRead*, Juelicher**);
void juelicher_fin(Juelicher*);
// end::mem[]

// tag::apply[]
void juelicher_apply(Juelicher*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]
/* end src/rbc/force/bending/juelicher/imp.h */
/* begin src/rbc/force/bending/juelicher/imp/type.h */
struct Juelicher {
    int4 *tri, *dih;

    float *area, *lentheta, *theta;
    float *lentheta_tot, *area_tot, *curva_mean_area_tot;
    float *f, *fad;

    int ne;
};
/* end src/rbc/force/bending/juelicher/imp/type.h */

/* begin src/mesh/force/kantor0/dev.h */
#ifdef FORCE_KANTOR0_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_kantor0_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor0_dev {
  #define END }
#endif

BEGIN

#ifdef FORCE_KANTOR0_HOST
_S_ double rsqrt0(double x) { return pow(x, -0.5); }
#define PRINT(fmt, ...) msg_print((fmt), ##__VA_ARGS__)
#define EXIT() ERR("assert")
#else
_S_ double rsqrt0(double x) { return rsqrt(x); }
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)
#endif

_S_ double max(double a, double b) { return a > b ? a : b; }

enum {ANGLE_OK, ANGLE_BAD_A, ANGLE_BAD_B};
_S_ int angle(double3 a, double3 b, /**/ double *pcos, double *pover_sin,
              double *pover_a, double *pover_b) {
    double  cost, over_sin, over_a, over_b;
    double aa, bb, ab;
    *pcos = *pover_sin = *pover_a = *pover_b = 0.0;

    aa = dot<double>(&a, &a);
    bb = dot<double>(&b, &b);
    ab = dot<double>(&a, &b);

    if (aa == 0) return ANGLE_BAD_A;
    if (bb == 0) return ANGLE_BAD_B;

    over_a = rsqrt0(aa);
    over_b = rsqrt0(bb);
    cost = ab * over_a * over_b;
    over_sin = rsqrt0(max(1.0 - cost*cost, 1e-6));

    *pcos = cost; *pover_sin = over_sin; *pover_a = over_a; *pover_b = over_b;
    return ANGLE_OK;
}

_S_ void dih_write(double3 a, double3 b, double3 c, double3 d) {
    PRINT("%g %g %g\n", a.x, a.y, a.z);
    PRINT("%g %g %g\n", b.x, b.y, b.z);
    PRINT("%g %g %g\n", c.x, c.y, c.z);
    PRINT("%g %g %g\n", d.x, d.y, d.z);
}

/* forces from one dihedral */
template <int update>
_S_ double3 dih0(double phi, double kb,
                 double3 a, double3 b, double3 c, double3 d) {
    int status;
    double over_k, over_n, cosTheta, over_sin,
        beta, b11, b12, sint0kb, cost0kb;
    double3 ab, ac, cd, bc, bd, da, k, n, k_n;
    diff(&a, &b, /**/ &ab);
    diff(&a, &c, /**/ &ac);
    diff(&c, &d, /**/ &cd);
    diff(&b, &d, /**/ &bd);
    diff(&b, &c, /**/ &bc);
    diff(&d, &a, /**/ &da);

    cross(&ab, &ac, /**/ &k);
    cross(&bd, &bc, /**/ &n);

    status = angle(k, n, /**/
                   &cosTheta, &over_sin,
                   &over_k, &over_n);
    if (status != ANGLE_OK) {
        PRINT("bad dihedral\n");
        dih_write(a, b, c, d);
        EXIT();
    }
    diff(&k, &n, /**/ &k_n);
    over_sin = copysign(over_sin, dot<double>(&k_n, &da));

    sint0kb = sin(phi) * kb;
    cost0kb = cos(phi) * kb;
    beta = cost0kb - cosTheta * sint0kb * over_sin;

    b11 = -beta *  cosTheta * over_k * over_k;
    b12 =  beta *             over_k * over_n;

    if (update == 1) {
        double3 r32, f1, f;
        diff(&c, &b, /**/ &r32);
        cross(&k, &r32, /**/ &f);
        cross(&n, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        double3 f, f1, f2, f3;
        double b22 = -beta * cosTheta * over_n * over_n;

        cross(&k, &ac, /**/ &f);
        cross(&k, &cd, /**/ &f1);
        cross(&n, &ac, /**/ &f2);
        cross(&n, &cd, /**/ &f3);

        scal(b11, /**/ &f);
        add(&f2, /**/ &f1);
        axpy(b12, &f1, /**/ &f);
        axpy(b22, &f3, /**/ &f);
        return f;
    }
    else {
        double3 f;
        f.x = f.y = f.z = 0;
        assert(0);
        return f;
    }
}

_I_ double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<1>(phi, kb, a, b, c, d);
}

_I_ double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<2>(phi, kb, a, b, c, d);
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
/* end src/mesh/force/kantor0/dev.h */

namespace juelicher_dev {
/* begin src/rbc/force/bending/juelicher/dev/main.h */
enum {X, Y, Z};
static __device__  int small(double s) {
    const double eps = 1e-12;
    if      (s >  eps) return 0;
    else if (s < -eps) return 0;
    else               return 1;
}
static __device__ void append(double x, int i, float *a) { atomicAdd(&a[i], x); }
static __device__ double tri_area(const double a[3], const double b[3], const double c[3]) { return tri_dev::kahan_area(a, b, c); }
static __device__ double tri_dih(const double a[3], const double b[3], const double c[3], const double d[3]) {
    double x, y;
    tri_dev::dihedral_xy(a, b, c, d, /**/ &x, &y);
    return -atan2(y, x); /* TODO: */
}
static __device__ void vec_scalar(const double a[3], double s, /**/ double b[3]) {
    b[X] = s*a[X]; b[Y] = s*a[Y]; b[Z] = s*a[Z];
}
static __device__ void vec_minus(const double a[3], const double b[3], /**/ double c[3]) {
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
static __device__ double vec_dot(const double a[3], const double b[3]) {
   return a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}
static __device__ void vec_cross(const double a[3], const double b[3], /**/ double c[3]) {
   c[X] = a[Y]*b[Z]-b[Y]*a[Z];
   c[Y] = b[X]*a[Z]-a[X]*b[Z];
   c[Z] = a[X]*b[Y]-b[X]*a[Y];
}
static __device__ double vec_abs(const double a[3]) { return sqrt(vec_dot(a, a)); }

static __device__ void vec_copy(const double a[3], double b[3]) {
    b[X] = a[X]; b[Y] = a[Y]; b[Z] = a[Z];
}

static __device__ double edg_abs(const double a[3], const double b[3]) {
    double u[3];
    vec_minus(b, a, u);
    return vec_abs(u);
}

static __device__ double tri_cot(const double a[3], const double b[3], const double c[3]) { /* at `b' */
    double x, y, u[3], v[3];
    y = 2 * tri_area(a, b, c);
    vec_minus(a, b, u);
    vec_minus(c, b, v);
    x = vec_dot(u, v);
    return x/y; /* TODO: */
}

static __device__ void vec_linear_combination(double al, const double a[3], double be, const double b[3],
                            /**/ double c[3]) {
    c[X] = al*a[X] + be*b[X];
    c[Y] = al*a[Y] + be*b[Y];
    c[Z] = al*a[Z] + be*b[Z];
}

static __device__ void vec_negative(const double a[3], /**/ double b[3]) {
    b[X] = -a[X]; b[Y] = -a[Y]; b[Z] = -a[Z];
}
static __device__ void vec_norm(const double a[3], /**/ double b[3]) {
    double s;
    s = vec_abs(a);
    if (!small(s)) vec_scalar(a, 1/s, /**/ b);
    else vec_copy(a, b);
}

static __device__ void dedg_abs(double a[3], double b[3], /**/ double da[3], double db[3]) {
    double u[3], e[3];
    vec_minus(b, a,   u);
    vec_norm(u,   e);
    vec_copy(e,     db);
    vec_negative(e, da);
}

static __device__ void vec_scalar_append(const double a[3], double s, int i,
                                         /**/ float *f) {
    append(s*a[X], 3*i,     f);
    append(s*a[Y], 3*i + 1, f);
    append(s*a[Z], 3*i + 2, f);
}

static __device__ void tri_normal(const double a[3], const double b[3], const double c[3], /**/ double e[3]) {
    double u[3], v[3], n[3];
    vec_minus(b, a,   u);
    vec_minus(c, a,   v);
    vec_cross(u, v,   n);
    vec_norm(n,   e);
}

static __device__ void ddih_angle(const double a[3], const double b[3], const double c[3], const double d[3], /**/
                                  double da[3], double db[3], double dc[3], double dd[3]) {
    double n[3], k[3];
    double e, An, Ak, cn, bn, bk, ck;

    tri_normal(a, b, c,   n);
    tri_normal(c, b, d,   k);

    e = edg_abs(c, b);

    An = tri_area(a, b, c);
    Ak = tri_area(c, b, d);

    vec_scalar(n, e/(2*An),   da);
    vec_scalar(k, e/(2*Ak),   dd);

    cn = tri_cot(b, c, a);
    bn = tri_cot(a, b, c);

    bk = tri_cot(c, b, d);
    ck = tri_cot(d, c, b);

    vec_linear_combination(-bn/e, n, -bk/e, k,    dc);
    vec_linear_combination(-cn/e, n, -ck/e, k,    db);
}

static __device__ void tri_edg(const double a[3], const double b[3], const double c[3],
                               /**/ double ab[3], double bc[3], double ca[3]) {
    vec_minus(b, a,   ab);
    vec_minus(c, b,   bc);
    vec_minus(a, c,   ca);
}

static __device__ void dtri_area(double a[3], double b[3], double c[3],  /**/ double da[3], double db[3], double dc[3]) {
    double n[3], n2[3], ab[3], bc[3], ca[3];
    tri_normal(a, b, c,   n);
    vec_scalar(n, 0.5,   n2);
    tri_edg(a, b, c,   ab, bc, ca);
    vec_cross(n2, bc,   da);
    vec_cross(n2, ca,   db);
    vec_cross(n2, ab,   dc);
}

static __device__ void get(const Particle *pp, int i, /**/ double r[3]) {
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}
static __device__ void get2(const Particle *pp, int i, int j,
                            /**/ double a[3], double b[3]) {
    get(pp, i, /**/ a);
    get(pp, j, /**/ b);
}
static __device__ void get3(const Particle *pp, int i, int j, int k,
                            /**/ double a[3], double b[3], double c[3]) {
    get(pp, i, /**/ a);
    get(pp, j, /**/ b);
    get(pp, k, /**/ c);
}
static __device__ void get4(const Particle *pp, int i, int j, int k, int l,
                            /**/ double a[3], double b[3], double c[3], double d[3]) {
    get(pp, i, /**/ a);
    get(pp, j, /**/ b);
    get(pp, k, /**/ c);
    get(pp, l, /**/ d);
}

__global__ void sum(int nv, const float *from, /**/ float *to) {
    int i, c;
    float s;
    i  = threadIdx.x + blockIdx.x * blockDim.x;
    c  = blockIdx.y;
    if (i < nv) s = from[c*nv + i]; else s = 0;
    s = warpReduceSum(s);
    if ((threadIdx.x % warpSize) == 0) atomicAdd(&to[c], s);
}

static __device__ void compute_area0(const Particle *pp, int4 tri, /**/ float *area) {
    double area0;
    double a[3], b[3], c[3];
    int i, j, k;
    i = tri.x; j = tri.y; k = tri.z;
    get3(pp, i, j, k, /**/ a, b, c);
    area0 = tri_area(a, b, c)/3;

    append(area0, i, area);
    append(area0, j, area);
    append(area0, k, area);
}

__global__ void compute_area(int nv, int nt, int nc,
                             const Particle *pp, const int4 *tri,
                             /**/ float *area) {
    int i;
    int t, c; /* triangle, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nt*nc) return;

    c = i / nt;
    t = i % nt;

    pp   += nv*c;
    area += nv*c;
    tri  += t;

    compute_area0(pp, *tri, /**/ area);
}

static __device__ void compute_theta_len0(const Particle *pp, int4 dih,
                                          /**/ float *theta, float *lentheta) {
    int i, j, k, l;
    double a[3], b[3], c[3], d[3], u[3];
    double len0, theta0, lentheta0;

    i = dih.x; j = dih.y; k = dih.z; l = dih.w;
    get4(pp, i, j, k, l, /**/ a, b, c, d);

    *theta = theta0 = tri_dih(a, b, c, d);
    vec_minus(c, b, u);
    len0 = vec_abs(u);
    lentheta0 = len0*theta0;

    append(lentheta0, j,  lentheta);
    append(lentheta0, k,  lentheta);
}

__global__ void compute_theta_len(int nv, int ne, int nc,
                                  const Particle *pp, const int4 *dih,
                                  /**/ float *theta, float *lentheta) {
    int i;
    int e, c; /* edge, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    theta    += ne*c + e;
    lentheta += nv*c;
    dih      += e;

    compute_theta_len0(pp, *dih, /**/ theta, lentheta);
}

__global__ void compute_mean_curv(int nc, float H0, float kad,
                                  const float *lentheta, const float *area,
                                  /**/ float *mean) {
    int i;
    float pi;
    pi = 3.141592653589793;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nc) return;
    mean[i] = (lentheta[i]/4 - H0 * area[i])*(4*kad*pi/area[i]);
}

static __device__ void force_edg0(float H0, float curva_mean_area_tot,
                                  const Particle *pp, const int4 dih,
                                  float theta, const float *lentheta, const float *area,
                                  /**/ float *f, float *fad) {
    int j, k;
    double b[3], c[3], db[3], dc[3];
    double coef;

    j = dih.y; k = dih.z;
    get2(pp, j, k, /**/ b, c);
    dedg_abs(b, c, db, dc);

    coef = - ( (lentheta[j]/area[j]/4 - H0) + (lentheta[k]/area[k]/4 - H0) ) * theta;
    vec_scalar_append(db, coef, j, f);
    vec_scalar_append(dc, coef, k, f);
    coef = -curva_mean_area_tot/4 * theta;
    vec_scalar_append(db, coef, j, fad);
    vec_scalar_append(dc, coef, k, fad);
}

__global__ void force_edg(int nv, int ne, int nc, float H0,
                          const Particle *pp, const int4 *dih,
                          const float *curva_mean_area_tot,
                          const float *theta, const float *lentheta, const float *area,
                          /**/ float *f, float *fad) {
    int i;
    int e, c; /* edge, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    dih      += e;

    curva_mean_area_tot += c;
    theta    += ne*c + e;
    lentheta += nv*c;
    area     += nv*c;

    f        += 3*nv*c;
    fad      += 3*nv*c;

    force_edg0(H0, *curva_mean_area_tot,
               pp, *dih,
               *theta, lentheta, area, /**/ f, fad);
}

static __device__ void force_lentheta0(float H0, float curva_mean_area_tot,
                                       const Particle *pp, const int4 dih,
                                       const float *lentheta, const float *area,
                                       /**/ float *f, float *fad) {
    int i, j, k, l;
    double len0, coef;
    double a[3], b[3], c[3], d[3], u[3];
    double da[3], db[3], dc[3], dd[3];

    i = dih.x; j = dih.y; k = dih.z; l = dih.w;
    get4(pp, i, j, k, l, /**/ a, b, c, d);
    ddih_angle(a, b, c, d, /**/ da, db, dc, dd);
    vec_minus(c, b, u);
    len0 = vec_abs(u);

    coef =  -(  (lentheta[j]/area[j]/4 - H0) + (lentheta[k]/area[k]/4 - H0) ) * len0 ;

    vec_scalar_append(da, coef, i, f);
    vec_scalar_append(db, coef, j, f);
    vec_scalar_append(dc, coef, k, f);
    vec_scalar_append(dd, coef, l, f);

    coef = -curva_mean_area_tot/2.0*len0;
    vec_scalar_append(da, coef, i, fad);
    vec_scalar_append(db, coef, j, fad);
    vec_scalar_append(dc, coef, k, fad);
    vec_scalar_append(dd, coef, l, fad);
}

__global__ void force_lentheta(int nv, int ne, int nc, float H0,
                               const Particle *pp, const int4 *dih,
                               const float *curva_mean_area_tot,
                               const float *lentheta, const float *area,
                               /**/ float *f, float *fad) {
    int i;
    int e, c; /* edge, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    dih      += e;

    curva_mean_area_tot += c;
    lentheta += nv*c;
    area     += nv*c;

    f        += 3*nv*c;
    fad      += 3*nv*c;

    force_lentheta0(H0, *curva_mean_area_tot,
                    pp, *dih,
                    lentheta, area, /**/ f, fad);
}

static __device__ void force_area0(float H0, const Particle *pp, const int4 tri,
                                   const float *lentheta, const float *area,
                                   /**/ float *f) {
    int i, j, k;
    double a[3], b[3], c[3];
    double da[3], db[3], dc[3];
    double coef1, coef2, coef;

    i = tri.x; j = tri.y; k = tri.z;
    get3(pp, i, j, k, /**/ a, b, c);

    dtri_area(a, b, c, da, db, dc);

    coef1 = 1.0/3.0;

    coef2 = lentheta[i]*lentheta[i]/8.0/area[i]/area[i] - 2.0*H0*H0;
    coef = coef1 * coef2;
    vec_scalar_append(da, coef, i, f);

    coef2 = lentheta[j]*lentheta[j]/8.0/area[j]/area[j] - 2.0*H0*H0;
    coef = coef1 * coef2;
    vec_scalar_append(db, coef, j, f);

    coef2 = lentheta[k]*lentheta[k]/8.0/area[k]/area[k] - 2.0*H0*H0;
    coef = coef1 * coef2;
    vec_scalar_append(dc, coef, k, f);
}

__global__ void force_area(int nv, int nt, int nc, float H0,
                           const Particle *pp, const int4 *tri,
                           const float *lentheta, const float *area,
                           /**/ float *f) {
    int i;
    int t, c; /* triangle, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nt*nc) return;

    c = i / nt;
    t = i % nt;

    pp       += nv*c;
    tri      += t;

    lentheta += nv*c;
    area     += nv*c;

    f        += 3*nv*c;

    force_area0(H0, pp, *tri,
                lentheta, area, /**/ f);
}

__global__ void accumulate(int n,
                           const float *ff0, const float *ff1,
                           /**/ Force *ff) {
    enum {X, Y, Z};
    int i;
    Force *f;
    const float *f0, *f1;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    f0 = &ff0[3*i];
    f1 = &ff1[3*i];
    f = &ff[i];

    f->f[X] += f0[X] + f1[X];
    f->f[Y] += f0[Y] + f1[Y];
    f->f[Z] += f0[Z] + f1[Z];
}
/* end src/rbc/force/bending/juelicher/dev/main.h */
}

/* begin src/rbc/force/bending/juelicher/imp/main.h */
void juelicher_ini(const MeshRead *cell, Juelicher **pq) {
#   define  nxt(h)     he_nxt(he, (h))
#   define  flp(h)     he_flp(he, (h))
#   define  ver(h)     he_ver(he, (h))
#   define  hdg_edg(e) he_hdg_edg(he, (e))
    Juelicher *q;
    int nt, nv, ne, e, i, j, k, l, h, n, nn, nnf;
    const int4 *tri;
    HeRead *he_read;
    He *he;
    int4 *dih;

    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tri = mesh_read_get_tri(cell);

    he_read_int4_ini(nv, nt, tri, /**/ &he_read);
    he_ini(he_read, /**/ &he);

    ne = he_ne(he);
    EMALLOC(ne, &dih);
    Dalloc(&q->dih, ne);
    Dalloc(&q->tri, nt);

    Dalloc(&q->theta,    MAX_CELL_NUM*ne); /* (sic) */
    Dalloc(&q->area,     MAX_CELL_NUM*nv);
    Dalloc(&q->lentheta, MAX_CELL_NUM*nv);

    Dalloc(&q->lentheta_tot, MAX_CELL_NUM);
    Dalloc(&q->area_tot,     MAX_CELL_NUM);
    Dalloc(&q->curva_mean_area_tot,     MAX_CELL_NUM);
    Dalloc(&q->f,   3*MAX_CELL_NUM*nv);
    Dalloc(&q->fad, 3*MAX_CELL_NUM*nv);

    for (e = 0; e < ne; e++) { /* i[jk]l */
        h = hdg_edg(e); n = nxt(h); nn = nxt(nxt(h));
        j = ver(h); k = ver(n); i = ver(nn);
        nnf = nxt(nxt(flp(h)));
        l = ver(nnf);
        dih[e].x = i; dih[e].y = j; dih[e].z = k; dih[e].w = l;
    }

    q->ne = ne;
    cH2D(q->dih, dih, ne);
    cH2D(q->tri, tri,  nt);

    he_read_fin(he_read);
    he_fin(he);
    EFREE(dih);

    *pq = q;
}

void juelicher_fin(Juelicher *q) {
    Dfree(q->dih);
    Dfree(q->tri);
    Dfree(q->area);
    Dfree(q->theta);
    Dfree(q->lentheta);
    Dfree(q->lentheta_tot);
    Dfree(q->area_tot);
    Dfree(q->curva_mean_area_tot);
    Dfree(q->f);
    Dfree(q->fad);
    EFREE(q);
}
/* end src/rbc/force/bending/juelicher/imp/main.h */
/* begin src/rbc/force/bending/juelicher/imp/forces.h */
/*
static void print(const char *s, float *dev) {
    float hst;
    cD2H(&hst, dev, 1);
    msg_print("%s: %g", s, hst);
    } */

/*
static void dump(int n, float *dev) {
    int i;
    float *hst;
    EMALLOC(n, &hst);
    cD2H(hst, dev, n);
    for (i = 0; i < n; i++)
        printf("%g\n", hst[i]);
    EFREE(hst);
    } */

/*
static void dump3(int n, float *dev) {
    int i, j;
    float *hst;
    EMALLOC(3*n, &hst);
    cD2H(hst, dev, 3*n);
    for (j = i = 0; i < n; i++) {
        printf("%g %g %g\n", hst[j], hst[j+1], hst[j+2]);
        j += 3;
    }
    EFREE(hst);
    } */

static void sum(int nv, int nc, const float *from, /**/ float *to) {
    dim3 thrd(128, 1);
    dim3 blck(ceiln(nv, thrd.x), nc);
    Dzero(to, nc);
    KL(juelicher_dev::sum, (blck, thrd), (nv, from, /**/ to));
}

void juelicher_apply(Juelicher *q, const RbcParams*, const RbcQuants *quants, /**/ Force *ff) {
    int nc, ne, nt, nv;
    //    RbcParams_v parv;

    int4 *tri, *dih;
    float *area, *lentheta, *theta;
    float *lentheta_tot, *area_tot, *curva_mean_area_tot;
    float *f, *fad;
    const Particle *pp;
    float pi, kb, H0, kad;
    pi = 3.141592653589793;

    //parv = rbc_params_get_view(par);

    H0 = -1.0/2.0;
    kb = 1.0; /* parv.kb; */
    kad = 2*kb/pi;

    tri = q->tri; dih = q->dih;
    area = q->area; lentheta = q->lentheta; theta = q->theta;
    lentheta_tot = q->lentheta_tot;
    area_tot = q->area_tot;
    curva_mean_area_tot = q->curva_mean_area_tot;
    f = q->f;
    fad = q->fad;
    pp = quants->pp;

    nv = quants->nv; nc = quants->nc; nt = quants->nt;
    ne = q->ne;

    if (!d::is_device_pointer(pp))
        ERR("`q->pp` is not a device pointer");
    if (quants->nc <= 0) return;

    // parv = rbc_params_get_view(par);

    Dzero(area, nv*nc);
    KL(juelicher_dev::compute_area, (k_cnf(nt*nc)), (nv, nt, nc, pp, tri, /**/ area));
    dSync();
    sum(nv, nc, area, /**/ area_tot); dSync();

    Dzero(theta, ne*nc);
    Dzero(lentheta, nv*nc);
    KL(juelicher_dev::compute_theta_len, (k_cnf(ne*nc)), (nv, ne, nc, pp, dih, /**/ theta, lentheta));
    dSync();

    sum(nv, nc, lentheta, /**/ lentheta_tot); dSync();
    KL(juelicher_dev::compute_mean_curv, (k_cnf(nc)), (nc, H0, kad, lentheta_tot, area_tot, /**/ curva_mean_area_tot));
    dSync();

    Dzero(f, nv*nc);
    Dzero(fad, nv*nc);
    KL(juelicher_dev::force_edg, (k_cnf(ne*nc)),
       (nv, ne, nc, H0, pp, dih, curva_mean_area_tot, theta, lentheta, area, /**/ f, fad));
    dSync();

    KL(juelicher_dev::force_lentheta, (k_cnf(ne*nc)),
       (nv, ne, nc, H0, pp, dih, curva_mean_area_tot, lentheta, area, /**/ f, fad));
    dSync();

    KL(juelicher_dev::force_area, (k_cnf(ne*nc)),
       (nv, nt, nc, H0, pp, tri, lentheta, area, /**/ f));
    dSync();

    KL(juelicher_dev::accumulate, (k_cnf(nv*nc)),
       (nv*nc, f, fad, /**/ ff));

    /*
    print("lentheta_tot", &lentheta_tot[0]);
    print("curva_mean_area_tot", &curva_mean_area_tot[0]);
    print("area", &area[0]);
    print("fx",  &f[0]);
    print("fad", &fad[0]);

    dump3(nv*nc, f); */
}
/* end src/rbc/force/bending/juelicher/imp/forces.h */
/* end src/rbc/force/bending/juelicher/imp.cu */

#line 1 "src/rbc/force/bending/kantor/imp.cu"

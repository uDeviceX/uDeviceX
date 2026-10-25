// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/mesh_bounce/imp.cu */
#include <assert.h>
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
/* begin src/d/api.h */
#ifndef UDX_MESH_DEVICE_API_COMMON
#define UDX_MESH_DEVICE_API_COMMON
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

#ifndef UDX_MESH_FORCE_KANTOR_COMMON_CUH
#define UDX_MESH_FORCE_KANTOR_COMMON_CUH
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
#endif

/* begin src/mesh_bounce/imp.h */
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

// tag::mem[]
void mesh_bounce_ini(int maxpp, /**/ MeshBB **d);
void mesh_bounce_fin(/**/ MeshBB *d);
// end::mem[]

// tag::int[]
void mesh_bounce_reini(int n, /**/ MeshBB *d);  // <1>
void mesh_bounce_find_collisions(float dt, int nm, MeshInfo meshinfo, const Positioncp *i_rr, int3 L,
                                 const int *starts, const int *counts, const Particle *pp, const Particle *pp0, /**/ MeshBB *d);  // <2>
void mesh_bounce_select_collisions(int n, /**/ MeshBB *d);  // <3>
void mesh_bounce_bounce(float dt, float mass, int n, const MeshBB *d, MeshInfo meshinfo,
                        const Positioncp *i_rr, const Particle *pp0, /**/ Particle *pp, Momentum *mm);  // <4>
// end::int[]

// tag::collect[]
void mesh_bounce_collect_rig_momentum(int ns, MeshInfo meshinfo, const Particle *pp, const Momentum *mm, /**/ Rigid *ss); // <1>
void mesh_bounce_collect_rbc_momentum(float mass, int nc, MeshInfo mi, const Momentum *mm, /**/ Particle *pp);            // <2>
// end::collect[]
/* end src/mesh_bounce/imp.h */
/* begin src/mesh_bounce/imp/type.h */
struct MeshBB {
    int *ncols;       /* number of possible collisions per particle      */
    float4 *datacol;  /* list of data related to collisions per particle */
    int *idcol;       /* list of triangle colliding ids per particle     */
};
/* end src/mesh_bounce/imp/type.h */

/* conf */
enum {MAX_COL = 4};

/* begin src/mesh_bounce/bbstates.h */
#define make_enum(a) a ,
#define make_str(a) #a ,

#define bbstates(_)                                                 \
    _(BB_SUCCESS)   /* succesfully bounced                       */ \
    _(BB_NOCROSS)   /* did not cross the plane                   */ \
    _(BB_WTRIANGLE) /* [w]rong triangle                          */ \
    _(BB_HFAIL)     /* no time solution h                        */ \
    _(BB_MAXCOL)    /* maximum number of collisions reached      */

enum BBState {
    bbstates(make_enum)
    BB_NSTATES
};

static const char *bbstatenames[] = {bbstates(make_str)};

#undef bbstates
#undef make_enum
#undef make_str
/* end src/mesh_bounce/bbstates.h */

#define _I_ static __device__
#define _S_ static __device__

namespace mesh_bounce_dev {
/* begin src/mesh_bounce/dev/type.h */
// typedef double   real_t;
// typedef double3 real3_t;

typedef float   real_t;
typedef float3 real3_t;

struct rPa {
    real3_t r, v;
};

_S_ real3_t make_real3(real_t a, real_t b, real_t c) {
    real3_t r;
    r.x = a;
    r.y = b;
    r.z = c;
    return r;
}

_I_ rPa fetch_Part(int i, const Particle *pp) {
    enum {X, Y, Z};
    const Particle *p = &pp[i];
    rPa rp = {
        .r = make_real3((real_t) p->r[X], (real_t) p->r[Y], (real_t) p->r[Z]),
        .v = make_real3((real_t) p->v[X], (real_t) p->v[Y], (real_t) p->v[Z])
    };
    return rp;
}

_I_ rPa fetch_Part(float dt, int i, const Positioncp *rr) {
    enum {X, Y, Z};
    const Positioncp r = rr[i];
    rPa rp;
    rp.r.x = r.rc[X];
    rp.r.y = r.rc[Y];
    rp.r.z = r.rc[Z];

    rp.v.x = (r.rc[X] - r.rp[X]) / dt;
    rp.v.y = (r.rc[Y] - r.rp[Y]) / dt;
    rp.v.z = (r.rc[Z] - r.rp[Z]) / dt;

    return rp;
}

_I_ void write_Part(const rPa *rp, int i, Particle *pp) {
    Particle p = {
        .r = {(float) rp->r.x, (float) rp->r.y, (float) rp->r.z},
        .v = {(float) rp->v.x, (float) rp->v.y, (float) rp->v.z}
    };
    pp[i] = p;
}
/* end src/mesh_bounce/dev/type.h */
/* begin src/mesh_bounce/dev/roots.h */
namespace roots {
#define SWAP(a,b) do { auto tmp = b ; b = a ; a = tmp ; } while(0)

template<typename real>
_I_ bool quadratic(real a, real b, real c, real *h0, real *h1) {
    const int sgnb = b > 0 ? 1 : -1;
    const real D = b*b - 4*a*c;

    if (D < 0) return false;

    *h0 = (-b - sgnb * sqrt(D)) / (2 * a);
    *h1 = c / (a * *h0);

    if (*h0 > *h1) SWAP(*h0, *h1);

    return true;
}

/* code copied from gsl library; gsl_poly_solve_cubic (see end of file for initial header)
   https://www.gnu.org/software/gsl/
   slightly adapted
*/

template <typename real>
_I_ int cubic(real a, real b, real c,
                     real *x0, real *x1, real *x2) {
    real q = (a * a - 3 * b);
    real r = (2 * a * a * a - 9 * a * b + 27 * c);

    real Q = q / 9;
    real R = r / 54;

    real Q3 = Q * Q * Q;
    real R2 = R * R;

    real CR2 = 729 * r * r;
    real CQ3 = 2916 * q * q * q;

    if (R == 0 && Q == 0) {
        *x0 = - a / 3 ;
        *x1 = - a / 3 ;
        *x2 = - a / 3 ;
        return 3 ;
    }
    else if (CR2 == CQ3) {
        /* this test is actually R2 == Q3, written in a form suitable
           for exact computation with integers */

        /* Due to finite precision some real roots may be missed, and
           considered to be a pair of complex roots z = x +/- epsilon i
           close to the real axis. */

        real sqrtQ = sqrt (Q);

        if (R > 0) {
            *x0 = -2 * sqrtQ  - a / 3;
            *x1 = sqrtQ - a / 3;
            *x2 = sqrtQ - a / 3;
        } else {
            *x0 = - sqrtQ  - a / 3;
            *x1 = - sqrtQ - a / 3;
            *x2 = 2 * sqrtQ - a / 3;
        }
        return 3 ;
    } else if (R2 < Q3) {
        real sgnR = (R >= 0 ? 1 : -1);
        real ratio = sgnR * sqrt (R2 / Q3);
        real theta = acos (ratio);
        real norm = -2 * sqrt (Q);
        *x0 = norm * cos (theta / 3) - a / 3;
        *x1 = norm * cos ((theta + 2.0 * M_PI) / 3) - a / 3;
        *x2 = norm * cos ((theta - 2.0 * M_PI) / 3) - a / 3;

        /* Sort *x0, *x1, *x2 into increasing order */

        if (*x0 > *x1)
            SWAP(*x0, *x1) ;

        if (*x1 > *x2) {
            SWAP(*x1, *x2) ;

            if (*x0 > *x1)
                SWAP(*x0, *x1) ;
        }

        return 3;
    } else {
        real sgnR = (R >= 0 ? 1 : -1);
        real A = -sgnR * pow (fabs (R) + sqrt (R2 - Q3), 1.0/3.0);
        real B = Q / A ;
        *x0 = A + B - a / 3;
        return 1;
    }
}

/* poly/solve_cubic.c
 *
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007, 2009 Brian Gough
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* solve_cubic.c - finds the real roots of x^3 + a x^2 + b x + c = 0 */
} // roots
/* end src/mesh_bounce/dev/roots.h */
/* begin src/mesh_bounce/dev/utils.h */
typedef double real;
typedef double3 real3;
enum {X, Y, Z};


template <typename T> _S_ T min3(T a, T b, T c) {return min(a, min(b, c));}
template <typename T> _S_ T max3(T a, T b, T c) {return max(a, max(b, c));}

template <typename T3>
_I_ T3 min3T3(T3 a, T3 b, T3 c) {
    T3 v;
    v.x = min3(a.x, b.x, c.x);
    v.y = min3(a.y, b.y, c.y);
    v.z = min3(a.z, b.z, c.z);
    return v;
}

template <typename T3>
_I_ T3 max3T3(T3 a, T3 b, T3 c) {
    T3 v;
    v.x = max3(a.x, b.x, c.x);
    v.y = max3(a.y, b.y, c.y);
    v.z = max3(a.z, b.z, c.z);
    return v;
}

_I_ int3 get_cidx(int3 L, real3_t r) {
    int3 c;
    c.x = floor((double) r.x + L.x/2);
    c.y = floor((double) r.y + L.y/2);
    c.z = floor((double) r.z + L.z/2);

    c.x = min(L.x-1, max(0, c.x));
    c.y = min(L.y-1, max(0, c.y));
    c.z = min(L.z-1, max(0, c.z));
    return c;
}


_I_ bool valid_time(real dt, real t) {return (t >= 0 && t <= dt);}

// BB assumes r0 + v0 dt = r1
_I_ void vprev(real dt, const real3_t *r1, const real3_t *r0, /**/ real3_t *v0) {
    v0->x = (r1->x - r0->x) / dt;
    v0->y = (r1->y - r0->y) / dt;
    v0->z = (r1->z - r0->z) / dt;
}

_I_ void fetch_triangle(int id, int nt, int nv, const int4 *tt, const Particle *i_pp,
                        /**/ rPa *A, rPa *B, rPa *C) {
    int4 t;
    int tid, mid;
    tid = id % nt;
    mid = id / nt;

    t = tt[tid];
    t.x += mid * nv;
    t.y += mid * nv;
    t.z += mid * nv;

    *A = fetch_Part(t.x, i_pp);
    *B = fetch_Part(t.y, i_pp);
    *C = fetch_Part(t.z, i_pp);
}

_I_ void fetch_triangle(float dt, int id, int nt, int nv, const int4 *tt, const Positioncp *i_rr,
                        /**/ rPa *A, rPa *B, rPa *C) {
    int4 t;
    int tid, mid;
    tid = id % nt;
    mid = id / nt;

    t = tt[tid];
    t.x += mid * nv;
    t.y += mid * nv;
    t.z += mid * nv;

    *A = fetch_Part(dt, t.x, i_rr);
    *B = fetch_Part(dt, t.y, i_rr);
    *C = fetch_Part(dt, t.z, i_rr);
}

_I_ void bounce_back(real dt, const rPa *p0, const real3_t *rw, const real3_t *vw, const real_t h, /**/ rPa *pn) {
    pn->v.x = 2 * vw->x - p0->v.x;
    pn->v.y = 2 * vw->y - p0->v.y;
    pn->v.z = 2 * vw->z - p0->v.z;

    pn->r.x = rw->x + (dt-h) * pn->v.x;
    pn->r.y = rw->y + (dt-h) * pn->v.y;
    pn->r.z = rw->z + (dt-h) * pn->v.z;
}

_I_ void lin_mom_change(float m, const real3_t v0, const real3_t v1, /**/ float dP[3]) {
    dP[X] = - m * (v1.x - v0.x);
    dP[Y] = - m * (v1.y - v0.y);
    dP[Z] = - m * (v1.z - v0.z);
}

_I_ void ang_mom_change(float m, real3_t rc, real3_t r0, real3_t v0, real3_t r1, real3_t v1, /**/ float dL[3]) {
    r0.x -= rc.x; r0.y -= rc.y; r0.z -= rc.z;
    r1.x -= rc.x; r1.y -= rc.y; r1.z -= rc.z;

    dL[X] = - m * (r1.y * v1.z - r1.z * v1.y  -  r0.y * v0.z + r0.z - v0.y);
    dL[Y] = - m * (r1.z * v1.x - r1.x * v1.z  -  r0.z * v0.x + r0.x - v0.z);
    dL[Z] = - m * (r1.x * v1.y - r1.y * v1.x  -  r0.x * v0.y + r0.y - v0.x);
}

/* shift origin from 0 to R for ang momentum */
_I_ void mom_shift_ref(const real3_t r, /**/ Momentum *m) {
    m->L[X] -= r.y * m->P[Z] - r.z * m->P[Y];
    m->L[Y] -= r.z * m->P[X] - r.x * m->P[Z];
    m->L[Z] -= r.x * m->P[Y] - r.y * m->P[X];
}

_S_ bool nz(float a) {return fabs(a) > 1e-6f;}

_I_ bool nonzero(const Momentum *m) {
    return nz(m->P[X]) && nz(m->P[Y]) && nz(m->P[Z]) &&
        nz(m->L[X]) && nz(m->L[Y]) && nz(m->L[Z]);
}

enum {XX, XY, XZ, YY, YZ, ZZ};
/* see /poc/bounce-back/trianglemom.mac */

/* inertia tensor w.r.t. com of triangle */
_S_ void compute_I(float mass, const real3_t a, const real3_t b, const real3_t c, /**/ real_t I[6]) {

    real3_t M, d, e;
    const real_t inv_12 = 1.f/12.f;
    const real_t inv_24 = 1.f/24.f;

    M.x = inv_12 * (a.x * a.x - a.x * b.x + b.x * b.x - c.x * (a.x + b.x) + c.x * c.x);
    M.y = inv_12 * (a.y * a.y - a.y * b.y + b.y * b.y - c.y * (a.y + b.y) + c.y * c.y);
    M.z = inv_12 * (a.z * a.z - a.z * b.z + b.z * b.z - c.z * (a.z + b.z) + c.z * c.z);

    d.x = 2 * a.x - b.x - c.x;
    d.y = 2 * b.x - a.x - c.x;
    d.z = 2 * c.x - a.x - b.x;

    e.x = 2 * a.y - b.y - c.y;
    e.y = 2 * b.y - a.y - c.y;
    e.z = 2 * c.y - a.y - b.y;

    I[XX] = mass * (M.y + M.z);
    I[YY] = mass * (M.x + M.z);
    I[ZZ] = mass * (M.x + M.y);

    I[XY] = - inv_24 * mass * (d.x * a.y + d.y * b.y + d.z * c.y);
    I[XZ] = - inv_24 * mass * (d.x * a.z + d.y * b.z + d.z * c.z);
    I[YZ] = - inv_24 * mass * (e.x * a.z + e.y * b.z + e.z * c.z);
}

_S_ void inverse(const real_t A[6], /**/ real_t I[6]) {

    /* minors */
    const real_t mx = A[YY] * A[ZZ] - A[YZ] * A[YZ];
    const real_t my = A[XY] * A[ZZ] - A[XZ] * A[YZ];
    const real_t mz = A[XY] * A[YZ] - A[XZ] * A[YY];

    /* inverse determinant */
    real_t idet = mx * A[XX] - my * A[XY] + mz * A[XZ];
    //assert( fabs(idet) > 1e-8f );
    idet = 1.f / idet;

    I[XX] =  idet * mx;
    I[XY] = -idet * my;
    I[XZ] =  idet * mz;
    I[YY] =  idet * (A[XX] * A[ZZ] - A[XZ] * A[XZ]);
    I[YZ] =  idet * (A[XY] * A[XZ] - A[XX] * A[YZ]);
    I[ZZ] =  idet * (A[XX] * A[YY] - A[XY] * A[XY]);
}

_S_ void rigid_motion_vel(const real3_t r, const real3_t om, const real3_t v0, /**/ real3_t *v) {
    v->x = v0.x + r.y * om.z - r.z * om.y;
    v->y = v0.y + r.z * om.x - r.x * om.z;
    v->z = v0.z + r.x * om.y - r.y * om.x;
}

_I_ void rbc_M2v(float mass, const Momentum m, real3_t a, real3_t b, real3_t c,
                 /**/ real3_t *va, real3_t *vb, real3_t *vc) {

    real_t I[6], Iinv[6], inv_mass;
    real3_t om, v, com;

    const real_t one_third = 1.f / 3.f;

    compute_I(mass, a, b, c, /**/ I);
    inverse(I, /**/ Iinv);

    /* angular velocity to be added (w.r.t. com of triangle) */
    om.x = one_third * (Iinv[XX] * m.L[X] + Iinv[XY] * m.L[Y] + Iinv[XZ] * m.L[Z]);
    om.y = one_third * (Iinv[XY] * m.L[X] + Iinv[YY] * m.L[Y] + Iinv[YZ] * m.L[Z]);
    om.z = one_third * (Iinv[XZ] * m.L[X] + Iinv[YZ] * m.L[Y] + Iinv[ZZ] * m.L[Z]);

    /* linear velocity to be added */
    inv_mass = 1.0 / mass;
    v.x =  one_third * m.P[X] * inv_mass;
    v.y =  one_third * m.P[Y] * inv_mass;
    v.z =  one_third * m.P[Z] * inv_mass;

    /* referential is com of triangle, shift it */
    com.x = one_third * (a.x + b.x + c.x);
    com.y = one_third * (a.y + b.y + c.y);
    com.z = one_third * (a.z + b.z + c.z);

    a.x -= com.x; b.x -= com.x; c.x -= com.x;
    a.y -= com.y; b.y -= com.y; c.y -= com.y;
    a.z -= com.z; b.z -= com.z; c.z -= com.z;

    rigid_motion_vel(a, om, v, va);
    rigid_motion_vel(b, om, v, vb);
    rigid_motion_vel(c, om, v, vc);
}
/* end src/mesh_bounce/dev/utils.h */
/* begin src/mesh_bounce/dev/intersection.h */
static __device__ bool cubic_root(real dt, real a, real b, real c, real d, /**/ real *h) {
    const real eps = 1e-8;
    real h1, h2, h3;

    if (fabs(a) > eps) { // cubic
        b /= a;
        c /= a;
        d /= a;

        int nsol = roots::cubic(b, c, d, &h1, &h2, &h3);

        if (valid_time(dt, h1))             {*h = h1; return true;}
        if (nsol > 1 && valid_time(dt, h2)) {*h = h2; return true;}
        if (nsol > 2 && valid_time(dt, h3)) {*h = h3; return true;}
    }
    else if (fabs(b) > eps) { // quadratic
        if (!roots::quadratic(b, c, d, &h1, &h2)) return false;
        if (valid_time(dt, h1)) {*h = h1; return true;}
        if (valid_time(dt, h2)) {*h = h2; return true;}
    }
    else if (fabs(c) > eps) { // linear
        h1 = -d/c;
        if (valid_time(dt, h1)) {*h = h1; return true;}
    }

    return false;
}

/* see Fedosov PhD Thesis */
_I_ BBState intersect_triangle(real dt,
                               const real3_t *s10, const real3_t *s20, const real3_t *s30,
                               const real3_t *vs1, const real3_t *vs2, const real3_t *vs3,
                               const rPa *p0,  /**/ real_t *h, real_t *u, real_t *v, real_t *s) {

    real3 a1, a2, at1, at2, n0, nt, ntt, dr0;

    diff(s20, s10, /**/ &a1);
    diff(s30, s10, /**/ &a2);

    diff(vs2, vs1, /**/ &at1);
    diff(vs3, vs1, /**/ &at2);

    /* n(t) = n + t*nt + t^2 * ntt */
    cross(&a1, &a2, /**/ &n0);
    cross(&at1, &at2, /**/ &ntt);

    nt.x = a1.y * at2.z - a1.z * at2.y  +  at1.y * a2.z - at1.z * a2.y;
    nt.y = a1.z * at2.x - a1.x * at2.z  +  at1.z * a2.x - at1.x * a2.z;
    nt.z = a1.x * at2.y - a1.y * at2.x  +  at1.x * a2.y - at1.y * a2.x;

    diff(&p0->r, s10, /**/ &dr0);

    /* check intersection with plane */
    {
        real3 r1, s11, n1, dr1;
        real b0, b1;

        apxb(&p0->r, dt, &p0->v, /**/ &r1);
        apxb(s10, dt, vs1, /**/ &s11);

        n1.x = n0.x + dt * (nt.x + dt * ntt.x);
        n1.y = n0.y + dt * (nt.y + dt * ntt.y);
        n1.z = n0.z + dt * (nt.z + dt * ntt.z);

        diff(&r1, &s11, /**/ &dr1);

        b0 = dot<real>(&dr0, &n0);
        b1 = dot<real>(&dr1, &n1);

        /* sign : which side does the particle belong? */
        *s = b0 > 0 ? 1 : -1;

        if (b0 * b1 > 0)
            return BB_NOCROSS;
    }

    /* find intersection time with plane */
    real hl;

    {
        real3 dv;
        real a, b, c, d;

        diff(&p0->v, vs1, /**/ &dv);

        a = dot<real>(&ntt, &dv);
        b = dot<real>(&ntt, &dr0) + dot<real>(&nt, &dv);
        c = dot<real>(&nt,  &dr0) + dot<real>(&n0, &dv);
        d = dot<real>(&n0, &dr0);

        if (!cubic_root(dt, a, b, c, d, &hl))
            return BB_HFAIL;
    }

    real3 rwl, g, a1_, a2_;
    real ga1, ga2, a11, a22, a12, fac, ul, vl;

    apxb(&p0->r, hl, &p0->v, /**/ &rwl);

    /* check if inside triangle */
    g.x = rwl.x - s10->x - hl * vs1->x;
    g.y = rwl.y - s10->y - hl * vs1->y;
    g.z = rwl.z - s10->z - hl * vs1->z;

    apxb(&a1, hl, &at1, /**/ &a1_);
    apxb(&a2, hl, &at2, /**/ &a2_);

    ga1 = dot<real>(&g, &a1_);
    ga2 = dot<real>(&g, &a2_);
    a11 = dot<real>(&a1_, &a1_);
    a12 = dot<real>(&a1_, &a2_);
    a22 = dot<real>(&a2_, &a2_);

    fac = 1.0 / (a11*a22 - a12*a12);

    ul = (ga1 * a22 - ga2 * a12) * fac;
    vl = (ga2 * a11 - ga1 * a12) * fac;

    if ((ul < 0) || (vl < 0) || (ul+vl > 1))
        return BB_WTRIANGLE;

    *h = hl;
    *u = ul;
    *v = vl;

    return BB_SUCCESS;
}
/* end src/mesh_bounce/dev/intersection.h */
/* begin src/mesh_bounce/dev/collect.h */
_S_ void rig_transform_mom(real3_t dr, Momentum *m) {
    mom_shift_ref(dr, /**/ m);

    m->L[X] += dr.y * m->P[Z] - dr.z * m->P[Y];
    m->L[Y] += dr.z * m->P[X] - dr.x * m->P[Z];
    m->L[Z] += dr.x * m->P[Y] - dr.y * m->P[X];
}

_S_ void rig_add_lin_mom(float mass, const float P[3], float *v) {
    enum {X, Y, Z};
    float fac = 1.0 / mass;
    atomicAdd(&v[X], fac * P[X]);
    atomicAdd(&v[Y], fac * P[Y]);
    atomicAdd(&v[Z], fac * P[Z]);
}

_S_ void rig_add_ang_mom(const float Iinv[], const float L[3], float *om) {
    enum {X, Y, Z, D};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};
    float dom[D] = {
        Iinv[XX] * L[X] + Iinv[XY] * L[Y] + Iinv[XZ] * L[Z],
        Iinv[YX] * L[X] + Iinv[YY] * L[Y] + Iinv[YZ] * L[Z],
        Iinv[ZX] * L[X] + Iinv[ZY] * L[Y] + Iinv[ZZ] * L[Z]
    };

    atomicAdd(&om[X], dom[X]);
    atomicAdd(&om[Y], dom[Y]);
    atomicAdd(&om[Z], dom[Z]);
}

/* assume very small portion of non zero momentum changes */
__global__ void collect_rig_mom(int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Rigid *ss) {
    int i, sid;
    Rigid *s;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    sid = i / nt;

    if (sid >= ns) return;

    Momentum m = mm[i];
    s = &ss[sid];

    if (nonzero(&m)) {
        rPa A, B, C;
        real3_t dr;

        fetch_triangle(i, nt, nv, tt, pp, /**/ &A, &B, &C);

        dr.x = s->com[X];
        dr.y = s->com[Y];
        dr.z = s->com[Z];

        dr.x -= 0.333333 * (A.r.x + B.r.x + C.r.x);
        dr.y -= 0.333333 * (A.r.y + B.r.y + C.r.y);
        dr.z -= 0.333333 * (A.r.z + B.r.z + C.r.z);

        rig_transform_mom(dr, &m);

        rig_add_lin_mom(s->mass, m.P, s->v);
        rig_add_ang_mom(s->Iinv, m.L, s->om);
    }
}

_S_ void addVel(const real3_t v, int i, Particle *pp) {
    enum {X, Y, Z};
    atomicAdd(pp[i].v + X, v.x);
    atomicAdd(pp[i].v + Y, v.y);
    atomicAdd(pp[i].v + Z, v.z);
}

__global__ void collect_rbc_mom(float mass, int nc, int nt, int nv, const int4 *tt, const Momentum *mm, /**/ Particle *pp) {
    int i, cid, tid;
    int4 t;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    tid = i % nt;
    cid = i / nt;

    if (cid >= nc) return;

    Momentum m = mm[i];

    if (nonzero(&m)) {
        rPa A, B, C;
        real3_t va, vb, vc;

        t = tt[tid];
        t.x += cid * nv;
        t.y += cid * nv;
        t.z += cid * nv;

        A = fetch_Part(t.x, pp);
        B = fetch_Part(t.y, pp);
        C = fetch_Part(t.z, pp);

        rbc_M2v(mass, m, A.r, B.r, C.r, /**/ &va, &vb, &vc);

        addVel(va, t.x, /**/ pp);
        addVel(vb, t.y, /**/ pp);
        addVel(vc, t.z, /**/ pp);
    }
}
/* end src/mesh_bounce/dev/collect.h */
/* begin src/mesh_bounce/dev/main.h */
static const real_t TOL = 0.1;

_S_ void get_cells(int3 L, float tol, real3_t A, real3_t B, real3_t C, /**/ int3 *lo, int3 *hi) {
    real3_t lf, hf;
    lf = min3T3(A, B, C);
    hf = max3T3(A, B, C);

    lf.x -= tol; lf.y -= tol; lf.z -= tol;
    hf.x += tol; hf.y += tol; hf.z += tol;

    *lo = get_cidx(L, lf);
    *hi = get_cidx(L, hf);
}

_S_ void find_collisions_cell(real dt, int tid, int start, int count, const Particle *pp, const Particle *pp0,
                              const rPa *A, const rPa *B, const rPa *C,
                              /**/ int *ncol, float4 *datacol, int *idcol) {
    int i, entry;
    rPa p, p0;
    BBState state;
    real_t tc, u, v, s;

    for (i = start; i < start + count; ++i) {
        p  = fetch_Part(i, pp);
        p0 = fetch_Part(i, pp0);

        vprev(dt, &p.r, &p0.r, /**/ &p0.v);

        state = intersect_triangle(dt, &A->r, &B->r, &C->r, &A->v, &B->v, &C->v, &p0, /**/ &tc, &u, &v, &s);

        if (state == BB_SUCCESS) {
            entry = atomicAdd(ncol + i, 1);

            if (entry >= MAX_COL) {
                state = BB_MAXCOL;
                break;
            }

            entry += i * MAX_COL;
            datacol[entry] = make_float4(tc, u, v, s);
            idcol[entry] = tid;
        }
    }
}

_S_ void revert_r(float h, rPa *p) {
    enum {X, Y, Z};
    p->r.x -= p->v.x * h;
    p->r.y -= p->v.y * h;
    p->r.z -= p->v.z * h;
}

__global__ void find_collisions(float dt, int nm, int nt, int nv, const int4 *tt, const Positioncp *i_rr,
                                int3 L, const int *starts, const int *counts, const Particle *pp, const Particle *pp0,
                                /**/ int *ncol, float4 *datacol, int *idcol) {

    int gid, ix, iy, iz, cid;
    rPa A, B, C;
    int3 str, end;
    gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= nm * nt) return;

    fetch_triangle(dt, gid, nt, nv, tt, i_rr, /**/ &A, &B, &C);

    revert_r(dt, &A);
    revert_r(dt, &B);
    revert_r(dt, &C);

    get_cells(L, TOL, A.r, B.r, C.r, /**/ &str, &end);

    for (iz = str.z; iz <= end.z; ++iz) {
        for (iy = str.y; iy <= end.y; ++iy) {
            for (ix = str.x; ix <= end.x; ++ix) {
                cid = ix + L.x * (iy + L.y * iz);

                find_collisions_cell(dt, gid, starts[cid], counts[cid], pp, pp0, &A, &B, &C, /**/ ncol, datacol, idcol);
            }
        }
    }
}

static const float MAX_T = 1e6f;

__global__ void select_collisions(int n, /**/ int *ncol, float4 *datacol, int *idcol) {
    int i, c, j, dst, src, argmin;
    float tmin;
    float4 d;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n) return;

    c = ncol[i];

    argmin = 0; tmin = MAX_T;
    for (j = 0; j < c; ++j) {
        src = MAX_COL * i + j;
        d = datacol[src];
        if (d.x < tmin) {
            argmin = j;
            tmin = d.x;
        }
    }

    if (c) {
        src = MAX_COL * i + argmin;
        dst = MAX_COL * i;
        idcol[dst]   = idcol[src];
        datacol[dst] = datacol[src];
        ncol[i] = 1; /* we can use find_collisions again */
    }
}

_S_ void get_collision_point(real dt,
                             const float4 dcol, rPa A, rPa B, rPa C,
                             /**/ real3_t *rw, real3_t *vw) {
    enum {X, Y, Z};
    float h, u, v, w;

    /* d.x is collision time */
    h = dt - dcol.x;
    revert_r(h, &A);
    revert_r(h, &B);
    revert_r(h, &C);

    u = dcol.y;
    v = dcol.z;
    w = 1 - u - v;

    rw->x = w * A.r.x + u * B.r.x + v * C.r.x;
    rw->y = w * A.r.y + u * B.r.y + v * C.r.y;
    rw->z = w * A.r.z + u * B.r.z + v * C.r.z;

    vw->x = w * A.v.x + u * B.v.x + v * C.v.x;
    vw->y = w * A.v.y + u * B.v.y + v * C.v.y;
    vw->z = w * A.v.z + u * B.v.z + v * C.v.z;
}

_S_ void push_particle(const real3_t *A, const real3_t *B, const real3_t *C, real_t l, /**/ real3_t *r) {
    real3_t n, a, b;
    real_t s;

    diff(B, A, /**/ &a);
    diff(C, A, /**/ &b);
    cross(&a, &b, /**/ &n);
    s = dot<real_t>(&n, &n);
    s = rsqrtf(s);
    n.x *= s;
    n.y *= s;
    n.z *= s;

    r->x += l*n.x;
    r->y += l*n.y;
    r->z += l*n.z;
}

static __device__ void atomicAddf3(float d[3], const float s[3]) {
    enum {X, Y, Z};
    atomicAdd(&d[X], s[X]);
    atomicAdd(&d[Y], s[Y]);
    atomicAdd(&d[Z], s[Z]);
}

__global__ void perform_collisions(float dt, float mass,
                                   int n, const int *ncol, const float4 *datacol, const int *idcol,
                                   int nt, int nv, const int4 *tt, const Positioncp *i_rr, const Particle *pp0,
                                   /**/ Particle *pp, Momentum *mm) {
    int i, id, entry;
    float4 d;
    rPa p1, p0, pn, A, B, C;
    real3_t rw, vw, rc;
    Momentum m;
    const real_t eps = 1e-2;

    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n || ncol[i] == 0) return;

    entry = i * MAX_COL;
    id = idcol[entry];
    d  = datacol[entry];

    p0 = fetch_Part(i, pp0);
    p1 = fetch_Part(i, pp);

    vprev(dt, &p1.r, &p0.r, /**/ &p0.v);

    fetch_triangle(dt, id, nt, nv, tt, i_rr, /**/ &A, &B, &C);

    get_collision_point(dt, d, A, B, C, /**/ &rw, &vw);

    bounce_back(dt, &p0, &rw, &vw, d.x, /**/ &pn);

    push_particle(&A.r, &B.r, &C.r, d.w * eps, /**/ &pn.r);

    write_Part(&pn, i, /**/ pp);

    /* add momentum (ang mom in ref of the triangle com) */
    /* shift in new ref */

    rc.x = 0.333333 * (A.r.x + B.r.x + C.r.x);
    rc.y = 0.333333 * (A.r.y + B.r.y + C.r.y);
    rc.z = 0.333333 * (A.r.z + B.r.z + C.r.z);

    lin_mom_change(mass,           p1.v,       pn.v, /**/ m.P);
    ang_mom_change(mass, rc, p1.r, p1.v, pn.r, pn.v, /**/ m.L);

    atomicAddf3(mm[id].P, m.P);
    atomicAddf3(mm[id].L, m.L);
}
/* end src/mesh_bounce/dev/main.h */
}

/* begin src/mesh_bounce/imp/main.h */
void mesh_bounce_ini(int maxpp, /**/ MeshBB **meshbb) {
    MeshBB *mbb;
    EMALLOC(1, meshbb);
    mbb = *meshbb;
    Dalloc(&mbb->ncols,   maxpp);
    Dalloc(&mbb->datacol, maxpp * MAX_COL);
    Dalloc(&mbb->idcol,   maxpp * MAX_COL);
}

void mesh_bounce_fin(/**/ MeshBB *mbb) {
    Dfree(mbb->ncols);
    Dfree(mbb->datacol);
    Dfree(mbb->idcol);
    EFREE(mbb);
}

void mesh_bounce_reini(int n, /**/ MeshBB *mbb) {
    CC(d::MemsetAsync(mbb->ncols, 0, n * sizeof(int)));
}

void mesh_bounce_find_collisions(float dt, int nm, MeshInfo mi, const Positioncp *i_rr, int3 L,
                                 const int *starts, const int *counts, const Particle *pp, const Particle *pp0,
                                 /**/ MeshBB *d) {
    if (!nm) return;
    KL(mesh_bounce_dev::find_collisions, (k_cnf(nm * mi.nt)),
       (dt, nm, mi.nt, mi.nv, mi.tt, i_rr, L, starts, counts, pp, pp0, /**/ d->ncols, d->datacol, d->idcol));
}

void mesh_bounce_select_collisions(int n, /**/ MeshBB *mbb) {
    KL(mesh_bounce_dev::select_collisions, (k_cnf(n)), (n, /**/ mbb->ncols, mbb->datacol, mbb->idcol));
}

void mesh_bounce_bounce(float dt, float mass, int n, const MeshBB *mbb, MeshInfo mi, const Positioncp *i_rr,
                   const Particle *pp0, /**/ Particle *pp, Momentum *mm) {
    KL(mesh_bounce_dev::perform_collisions, (k_cnf(n)),
       (dt, mass, n, mbb->ncols, mbb->datacol, mbb->idcol, mi.nt, mi.nv, mi.tt, i_rr, pp0, /**/ pp, mm));
}

void mesh_bounce_collect_rig_momentum(int ns, MeshInfo mi, const Particle *pp, const Momentum *mm, /**/ Rigid *ss) {
    KL(mesh_bounce_dev::collect_rig_mom, (k_cnf(ns * mi.nt)), (ns, mi.nt, mi.nv, mi.tt, pp, mm, /**/ ss));
}

void mesh_bounce_collect_rbc_momentum(float mass, int nc, MeshInfo mi, const Momentum *mm, /**/ Particle *pp) {
    KL(mesh_bounce_dev::collect_rbc_mom, (k_cnf(nc * mi.nt)), (mass, nc, mi.nt, mi.nv, mi.tt, mm, /**/ pp));
}
/* end src/mesh_bounce/imp/main.h */

#undef _I_
#undef _S_
/* end src/mesh_bounce/imp.cu */

#line 1 "src/mpi/glb.cpp"

// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/clist/imp.cu */
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
#ifndef UDX_DBG_CLIST_MSG_COMMON
#define UDX_DBG_CLIST_MSG_COMMON
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
#endif
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

/* begin src/inc/def.h */
#ifndef UDX_DBG_CLIST_DEF_COMMON
#define UDX_DBG_CLIST_DEF_COMMON
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
#ifndef UDX_DBG_CLIST_TYPE_COMMON
#define UDX_DBG_CLIST_TYPE_COMMON
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
/* begin src/algo/scan/imp.h */
struct Scan;

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
// end::interface[]
/* end src/algo/scan/imp.h */

/* begin src/d/api.h */
#ifndef UDX_DBG_CLIST_D_API_COMMON
#define UDX_DBG_CLIST_D_API_COMMON
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
#ifndef UDX_DBG_CLIST_KL_COMMON
#define UDX_DBG_CLIST_KL_COMMON
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

/* begin src/struct/partlist/type.h */
#ifndef UDX_CLIST_PARTLIST_COMMON
#define UDX_CLIST_PARTLIST_COMMON
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
#endif
/* begin src/struct/partlist/dev.h */
static __device__ bool is_dead(int i, const PartList lp) {
    if (lp.deathlist)
        return lp.deathlist[i] == DEAD;
    return false;
}

static __device__ bool is_alive(int i, const PartList lp) {
    return !is_dead(i, lp);
}
/* end src/struct/partlist/dev.h */

/* begin src/struct/particle/dev.h */
static __device__ float3 fetch_pos(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 r;
    r.x = pp[i].r[X]; r.y = pp[i].r[Y]; r.z = pp[i].r[Z];
    return r;
}

static __device__ float3 fetch_vel(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 v;
    v.x = pp[i].v[X]; v.y = pp[i].v[Y]; v.z = pp[i].v[Z];
    return v;
}
/* end src/struct/particle/dev.h */

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
/* begin src/clist/dev.h */
#ifndef UDX_CLIST_DEV_COMMON
#define UDX_CLIST_DEV_COMMON
#define _I_ static __device__

enum { CLIST_NMAX = (256*256*256) };

// tag::int[]
_I_ uint clist_encode_id(int s, int pid)
// end::int[]
{
    /* pid: particle id; s: array id */
    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

// tag::int[]
_I_ void clist_decode_id(uint c, /**/ int *s, int *pid)
// end::int[]
{
    /* pid: particle id; s: array id; c : code */
    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}

#undef _I_
/* end src/clist/dev.h */
#endif

/* begin src/clist/imp/type.h */
// tag::clist[]
struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};
// end::clist[]

// tag::map[]
struct ClistMap {
    int nA;       /* number of source arrays to build the cell lists, e.g remote+bulk -> 2 */
    uchar4 **ee;  /* cell entries */
    uint *ii;     /* codes containing: indices of data to fetch and array id from which to fetch */
    Scan *scan;   /* scan workspace */
    long maxp;    /* maximum number of particles per input vector */
};
// end::map[]
/* end src/clist/imp/type.h */
namespace clist_dev {
/* begin src/clist/dev/main.h */
enum {
    VALID   = 0,
    INVALID = 255
};

static __device__ bool inside(float3 r, int3 L) {
    return
        (r.x >= -L.x/2) && (r.x < L.x/2) &&
        (r.y >= -L.y/2) && (r.y < L.y/2) &&
        (r.z >= -L.z/2) && (r.z < L.z/2)  ;
}

static __device__ int project_cid(int i, const int L) {
    return i < 0 ? 0 : i >= L ? L - 1 : i;
}

static __device__ uchar4 get_entry(const bool project, float3 r, int3 L) {
    uchar4 e;
    int ix, iy, iz;

    /* must be done in double precision */
    ix = (int) ((double) r.x + L.x/2);
    iy = (int) ((double) r.y + L.y/2);
    iz = (int) ((double) r.z + L.z/2);

    if (project) {
        ix = project_cid(ix, L.x);
        iy = project_cid(iy, L.y);
        iz = project_cid(iz, L.z);
    }

    if (project || inside(r, L)) e.w =   VALID;
    else                         e.w = INVALID;

    e.x = ix; e.y = iy; e.z = iz;
    return e;
}

static __device__ int get_cid(int3 L, uchar4 e) {
    return e.x + L.x * (e.y + L.y * e.z);
}

/* project:: if particle is outside of domain L, include it in closest cell */
__global__ void subindex(bool project, int3 L, int n, const PartList lp, /*io*/ int *counts, /**/ uchar4 *ee) {
    int i, cid;
    float3 r;
    uchar4 e;
    bool dead;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    dead = is_dead(i, lp);
    r = fetch_pos(i, lp.pp);

    if (dead) {
        e.x = e.y = e.z = 0;
        e.w = INVALID;
    } else {
        e = get_entry(project, r, L);
        cid = get_cid(L, e);
    }

    if (e.w != INVALID)
        e.w = atomicAdd(counts + cid, 1);

    ee[i] = e;
}

__global__ void get_ids(int array_id, int3 L, int n, const int *starts, const uchar4 *ee, /**/ uint *ii) {
    int i, cid, id, start;
    uchar4 e;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    e = ee[i];
    cid = get_cid(L, e);

    if (e.w != INVALID) {
        start = starts[cid];
        id = start + e.w;
        ii[id] = clist_encode_id(array_id, i);
    }
}

template <typename T, int N>
__device__ void fetch(const Sarray<const T*, N> src, uint i, /**/ T *d) {
    int src_id, array_id;
    clist_decode_id(i, /**/ &array_id, &src_id);
    *d = src.d[array_id][src_id];
}

template <typename T, int N>
__global__ void gather(const Sarray<const T*, N> src, const uint *ii, int n, /**/ T *dd) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    uint code = ii[i];
    fetch(src, code, /**/ dd + i);
}
/* end src/clist/dev/main.h */
}
/* begin src/clist/imp/ini.h */
void clist_ini(int LX, int LY, int LZ, /**/ Clist **cl) {
    Clist *c;
    EMALLOC(1, cl);
    c = *cl;
    c->dims.x = LX;
    c->dims.y = LY;
    c->dims.z = LZ;
    c->ncells = LX * LY * LZ;

    int n = (c->ncells + 16);
    Dalloc(&c->starts, n);
    Dalloc(&c->counts, n);
}

void clist_ini_map(int maxp, int nA, const Clist *c, /**/ ClistMap **map) {
    long i, size;
    EMALLOC(1, map);
    ClistMap *m = *map;

    UC(scan_ini(c->ncells + 16, /**/ &m->scan));

    m->maxp = maxp;

    m->nA = nA;
    EMALLOC(nA, &m->ee);

    for (i = 0; i < nA; ++i) Dalloc(&m->ee[i], maxp);

    size = nA * maxp;
    Dalloc(&m->ii, size);
}
/* end src/clist/imp/ini.h */
/* begin src/clist/imp/fin.h */
void clist_fin(/**/ Clist *c) {
    Dfree(c->starts);
    Dfree(c->counts);
    EFREE(c);
}

void clist_fin_map(ClistMap *m) {
    UC(scan_fin(/**/ m->scan));

    for (int i = 0; i < m->nA; ++i) Dfree(m->ee[i]);
    EFREE(m->ee);
    Dfree(m->ii);
    EFREE(m);
}
/* end src/clist/imp/fin.h */
/* begin src/clist/imp/main.h */
enum {LOCAL, REMOTE};

const int* clists_get_ss(const Clist *c) {return c->starts;}
const int* clists_get_cc(const Clist *c) {return c->counts;}
int3       clists_get_dim(const Clist *c)  {return c->dims;}
int        clists_get_n(const Clist *c)  {return c->ncells;}
const uint* clist_get_ids(const ClistMap *m) {return m->ii;}

void clist_ini_counts(/**/ Clist *c) {
    if (c->ncells) CC(d::MemsetAsync(c->counts, 0, (c->ncells + 16) * sizeof(int)));
}

static void check_input_size(int id, long n, const ClistMap *m) {
    long cap = m->maxp;
    if (n > cap)
        ERR("Too many input particles for array %d (%ld / %ld)", id, n, cap);
}

static void comp_subindices(bool project, int n, const PartList lp, int3 dims, /**/ int *cc, uchar4 *ee) {
    if (n) KL(clist_dev::subindex, (k_cnf(n)), (project, dims, n, lp, /*io*/ cc, /**/ ee));
}

void clist_subindex(bool project, int aid, int n, const PartList lp, /**/ Clist *c, ClistMap *m) {
    UC(check_input_size(aid, n, m));
    UC(comp_subindices(project, n, lp, c->dims, /**/ c->counts, m->ee[aid]));
}

void clist_subindex_local(int n, const PartList lp, /**/ Clist *c, ClistMap *m) {
    UC(clist_subindex(false, LOCAL, n, lp, /**/ c, m));
}

void clist_subindex_remote(int n, const PartList lp, /**/ Clist *c, ClistMap *m) {
    UC(clist_subindex(false, REMOTE, n, lp, /**/ c, m));
}

void clist_build_map(const int nn[], /**/ Clist *c, ClistMap *m) {
    int nc, *cc, *ss, n, i;
    const uchar4 *ee;
    uint *ii = m->ii;
    int3 dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;

    scan_apply(cc, nc + 16, /**/ ss, /*w*/ m->scan);

    for (i = 0; i < m->nA; ++i) {
        n = nn[i];
        ee = m->ee[i];
        UC(check_input_size(i, n, m));
        if (n) KL(clist_dev::get_ids, (k_cnf(n)), (i, dims, n, ss, ee, /**/ ii));
    }
}

static void check_map_capacity(long n, const ClistMap *m) {
    long cap = m->maxp * m->nA;
    if (n > cap)
        ERR("Too many particles for this cell list (%ld / %ld)", n, cap);
}

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout, /**/ Particle *ppout) {
    Sarray <const Particle*, 2> src = {pplo, ppre};
    UC(check_map_capacity(nout, m));
    if (nout)
        KL(clist_dev::gather, (k_cnf(nout)), (src, m->ii, nout, /**/ ppout));
}

void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout, /**/ int *iiout) {
    Sarray <const int*, 2> src = {iilo, iire};
    UC(check_map_capacity(nout, m));
    if (nout)
        KL(clist_dev::gather, (k_cnf(nout)), (src, m->ii, nout, /**/ iiout));
}

void clist_build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, ClistMap *m) {
    const int nn[] = {nlo, 0};
    PartList lp;
    lp.pp = pplo;
    lp.deathlist = NULL;
    UC(clist_ini_counts(/**/ c));
    UC(clist_subindex_local (nlo, lp, /**/ c, m));
    UC(clist_build_map(nn, /**/ c, m));
    UC(clist_gather_pp(pplo, NULL, m, nout, ppout));
}
/* end src/clist/imp/main.h */
/* end src/clist/imp.cu */

#line 1 "src/cnt/imp.cu"

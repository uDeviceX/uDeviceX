// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/distr/flu/imp.cu */
#include <mpi.h>
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
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif /* UDX_DISTR_EXCH_KL_COMMON */
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

/* begin src/d/api.h */
#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
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
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
/* begin src/inc/type.h */
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
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
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
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
/* begin src/frag/dev.h */
#ifndef UDX_DISTR_EXCH_FRAG_DEV_COMMON
#define UDX_DISTR_EXCH_FRAG_DEV_COMMON
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
/* end src/frag/dev.h */
#endif /* UDX_DISTR_EXCH_FRAG_DEV_COMMON */
/* begin src/frag/imp.h */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
enum {frag_bulk = 26};

struct int3;

namespace frag_hst {

// tag::i2d[]
int i2dx(int i);                 // <1>
int i2dy(int i);                 // <2>
int i2dz(int i);                 // <3>
void i2d3(int i, /**/ int d[3]); // <4>
// end::i2d[]

// tag::d2i[]
int d2i(int x, int y, int z); // <1>
int d32i(const int d[3]);     // <2>
// end::d2i[]

// tag::ncell[]
int ncell(int3 L, int i); // <1>
// end::ncell[]

// tag::anti[]
int antid2i(int x, int y, int z); // <1>
int anti(int i);                  // <2>
// end::anti[]

// tag::hst[]
void estimates(int3 L, int nfrags, float maxd, /**/ int *cap); // <1>
// end::hst[]

} // frag_hst
/* end src/frag/imp.h */
#endif /* UDX_DISTR_EXCH_FRAG_IMP_COMMON */

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

/* begin src/algo/scan/imp.h */
struct Scan;

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
// end::interface[]
/* end src/algo/scan/imp.h */
/* begin src/algo/utils/shfl.h */
#ifndef UDX_DISTR_EXCH_SHFL_COMMON
#define UDX_DISTR_EXCH_SHFL_COMMON
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
#endif /* UDX_DISTR_EXCH_SHFL_COMMON */

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
/* begin src/flu/imp.h */
// tag::struct[]
struct FluQuants {
    Particle *pp, *pp0;    /* particles on device  */
    int       n;           /* particle number      */
    Clist     *cells;      /* cell lists           */
    ClistMap *mcells;      /* cell lists map       */
    Particle *pp_hst;      /* particles on host    */

    /* optional data */

    bool ids, colors;

    int *ii, *ii0;  /* global ids on device */
    int *ii_hst;    /* global ids on host   */

    int *cc, *cc0;  /* colors on device */
    int *cc_hst;    /* colors on host   */

    int maxp; /* maximum particle number */
};
// end::struct[]

struct Coords;
struct GenColor;

// tag::mem[]
void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q);
void flu_fin(FluQuants *q);
// end::mem[]

// tag::gen[]
void flu_gen_quants(const Coords*, int numdensity, const GenColor *gc, FluQuants *q); // <1>
void flu_gen_ids(MPI_Comm, const int n, FluQuants *q); // <2>
// end::gen[]

// tag::start[]
void flu_strt_quants(MPI_Comm, const char *base, const int id, FluQuants *q);     // <1>
void flu_strt_dump(MPI_Comm, const char *base, const int id, const FluQuants *q); // <2>
// end::start[]

// tag::tools[]
void flu_txt_dump(const Coords*, const FluQuants *q); // <1>

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void flu_build_cells(/**/ FluQuants *q); // <2>
// end::tools[]
/* end src/flu/imp.h */

/* begin src/comm/imp.h */
#ifndef UDX_DISTR_EXCH_COMM_COMMON
#define UDX_DISTR_EXCH_COMM_COMMON
typedef unsigned char data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

/* Allocation mod for bags */
enum AllocMod {
    // tag::AllocMod[]
    HST_ONLY,   /* only host bags allocated                 */
    DEV_ONLY,   /* only device bags allocated               */
    PINNED,     /* both host and device pinned              */
    PINNED_HST, /* host pinned; no device memory            */
    PINNED_DEV, /* host pinned; device global memory on gpu */
    NONE        /* no allocation                            */
    // end::AllocMod[]
};

// tag::dBags[]
struct dBags {
    data_t *data[NBAGS]; /* data on the device         */
};
// end::dBags[]

// tag::hBags[]
struct hBags {
    data_t *data[NBAGS]; /* data on the host                    */
    int         *counts; /* size of the data                    */
    int capacity[NBAGS]; /* capacity of each frag (elem number) */
    size_t bsize;        /* size of one datum in bytes          */
};
// end::hBags[]

struct Comm;
struct CommBuffer;

// tag::alloc[]
void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db);
void comm_bags_fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db);
void comm_ini(MPI_Comm cart, /**/ Comm **c);
void comm_fin(/**/ Comm *c);
void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer**);
void comm_buffer_fin(CommBuffer*);
// end::alloc[]


void comm_buffer_set(int nbags, const hBags*, CommBuffer*);
void comm_buffer_get(const CommBuffer*, int nbags, hBags*);

// tag::communication[]
void comm_post_recv(hBags *b, Comm *c);           // <1>
void comm_post_send(const hBags *b, Comm *c);     // <2>

void comm_wait_recv(Comm *c, /**/ hBags *b);      // <3>
void comm_wait_send(Comm *c);                     // <4>
// end::communication[]

// tag::communication_b[]
void comm_post_recv(CommBuffer *cb, Comm *c);
void comm_post_send(const CommBuffer *cb, Comm *c);

void comm_wait_recv(Comm *c, /**/ CommBuffer *cb);
void comm_wait_send(Comm *c);
// end::communication_b[]

// tag::get[]
int    comm_get_number_capacity(int i, const hBags *b); // <1>
size_t comm_get_byte_capacity(int i, const hBags *b);   // <2>
// end::get[]

/* end src/comm/imp.h */
#endif /* UDX_DISTR_EXCH_COMM_COMMON */
/* begin src/comm/utils.h */
#ifndef UDX_DISTR_EXCH_COMM_UTILS_COMMON
#define UDX_DISTR_EXCH_COMM_UTILS_COMMON
// tag::convert[]
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
// end::convert[]
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
/* end src/comm/utils.h */
#endif /* UDX_DISTR_EXCH_COMM_UTILS_COMMON */

/* begin src/distr/map/type.h */
#ifndef UDX_DISTR_EXCH_DMAP_TYPE_COMMON
#define UDX_DISTR_EXCH_DMAP_TYPE_COMMON
// tag::struct[]
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
// end::struct[]
/* end src/distr/map/type.h */
#endif /* UDX_DISTR_EXCH_DMAP_TYPE_COMMON */
/* begin src/distr/flu/imp.h */
struct DFluPack;
struct DFluComm;
struct DFluUnpack;
struct DFluStatus;

// tag::interface[]
void dflu_pack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluPack **p);
void dflu_comm_ini(MPI_Comm comm, /**/ DFluComm **c);
void dflu_unpack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluUnpack **u);

void dflu_pack_fin(DFluPack *p);
void dflu_comm_fin(DFluComm *c);
void dflu_unpack_fin(DFluUnpack *u);

/* map */
void dflu_build_map(int n, const PartList lp, DFluPack *p);

/* pack */
void dflu_pack(const FluQuants *q, /**/ DFluPack *p);
void dflu_download(DFluPack *p, /**/ DFluStatus *s);

/* communication */
void dflu_post_recv(DFluComm *c, DFluUnpack *u);
void dflu_post_send(DFluPack *p, DFluComm   *c);
void dflu_wait_recv(DFluComm *c, DFluUnpack *u);
void dflu_wait_send(DFluComm *c);

/* unpack */
void dflu_unpack(/**/ DFluUnpack *u);
// end::interface[]


// tag::clist[]
/* cell lists */
void dflu_bulk(PartList lp, /**/ FluQuants *q);
void dflu_halo(const DFluUnpack *u, /**/ FluQuants *q);
void dflu_gather(int ndead, const DFluPack *p, const DFluUnpack *u, /**/ FluQuants *q);
// end::clist[]
/* end src/distr/flu/imp.h */

/* begin src/distr/map/dev.h */
#ifndef UDX_DISTR_EXCH_DMAP_DEV_COMMON
#define UDX_DISTR_EXCH_DMAP_DEV_COMMON
// tag::int[]
/* exclusive scan */
template <int NCOUNTS>
static __global__ void dmap_scan(/**/ DMap m)              // <1>
// end::int[]
{
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0, cnt = 0;

    if (tid < NCOUNTS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * shfl_up(val, L) ;
    if (tid <= NCOUNTS) m.starts[tid] = val - cnt;
}

// tag::int[]
static __device__ int dmap_get_fid(int3 L, const float r[3])       // <2>
// end::int[]
{
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -L.x/2) + (r[X] >= L.x/2);
    y = -1 + (r[Y] >= -L.y/2) + (r[Y] >= L.y/2);
    z = -1 + (r[Z] >= -L.z/2) + (r[Z] >= L.z/2);
    return frag_dev::d2i(x, y, z);
}

// tag::int[]
static __device__ void dmap_add(int pid, int fid, DMap m)  // <3>
// end::int[]
{
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}
/* end src/distr/map/dev.h */
#endif /* UDX_DISTR_EXCH_DMAP_DEV_COMMON */
/* begin src/distr/map/imp.h */
// tag::int[]
void dmap_ini(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin(int nfrags, /**/ DMap *m);
void dmap_reini(int nfrags, /**/ DMap m);
void dmap_download_counts(int nfrags, /**/ DMap *m);
void dmap_ini_host(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin_host(int nfrags, /**/ DMap *m);
void dmap_reini_host(int nfrags, /**/ DMap m);
void dmap_D2H(int nfrags, const DMap *d, /**/ DMap *h);
// end::int[]
/* end src/distr/map/imp.h */
/* begin src/distr/common/imp.h */
struct Particle;
struct int3;

// tag::int[]
void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf); // <1>
void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp); // <3>
// end::int[]

/* end src/distr/common/imp.h */
/* begin src/distr/flu/status/imp.h */
struct DFluStatus;
struct Config;

/* ini */
void dflu_status_ini(DFluStatus **s);
void dflu_status_fin(DFluStatus  *s);

/* get */
int  dflu_status_nullp(DFluStatus *s);
int  dflu_status_success(DFluStatus *s);
void dflu_status_log(DFluStatus *s);

/* set */
void dflu_status_exceed(int frag, int counter, int capacity, /**/ DFluStatus *s);

/* end src/distr/flu/status/imp.h */

/* begin src/distr/flu/imp/type.h */
typedef Sarray <float2*, 26> float2p26;
typedef Sarray <int*, 26> intp26;

enum {
    MAX_NBAGS = 3
};

// tag::struct[]
struct DFluPack {
    DMap map;
    dBags dbags[MAX_NBAGS], *dpp, *dii, *dcc;
    hBags hbags[MAX_NBAGS], *hpp, *hii, *hcc;
    int nbags;
    int nhalo; /* number of sent particles */
    int3 L; /* subdomain size */
    CommBuffer *hbuf;
};

struct DFluComm {
    Comm *comm;
};

struct DFluUnpack {
    hBags hbags[MAX_NBAGS], *hpp, *hii, *hcc;
    int nbags;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo; /* number of received particles */
    int3 L; /* subdomain size */
    CommBuffer *hbuf;
};
// end::struct[]
/* end src/distr/flu/imp/type.h */

namespace dflu_dev {
/* begin src/distr/flu/dev.h */
__global__ void build_map(int3 L, const PartList lp, const int n, /**/ DMap m) {
    int pid, fid;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    if (is_dead(pid, lp)) return;

    const Particle p = lp.pp[pid];

    fid = dmap_get_fid(L, p.r);

    if (fid != frag_bulk)
        dmap_add(pid, fid, /**/ m);
}

template <typename T, int STRIDE>
__global__ void pack(const T *data, DMap m, /**/ Sarray<T*, 26> buf) {
    int gid, slot;
    int fid; /* [f]ragment [id] */
    int offset, pid, c, d, s;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    slot = gid / STRIDE;
    fid = frag_dev::frag_get_fid(m.starts, slot);
    if (slot >= m.starts[26]) return;
    c = gid % STRIDE;

    offset = slot - m.starts[fid];
    pid = __ldg(m.ids[fid] + offset);

    d = c + STRIDE * offset;
    s = c + STRIDE * pid;

    buf.d[fid][d] = data[s];
}
/* end src/distr/flu/dev.h */
}
/* begin src/distr/flu/imp/ini.h */
static int estimate_max_flux(int3 L, int fid, int maxd) {
    int e, nfaces, d[3];
    frag_hst::i2d3(fid, d);
    nfaces = abs(d[0]) + abs(d[1]) + abs(d[2]);
    e = maxd * frag_hst::ncell(L, fid) * nfaces;
    return e;
}

static void get_capacity(int3 L, int maxd, /**/ int capacity[NBAGS]) {
    for (int i = 0; i < NFRAGS; ++i)
        capacity[i] = estimate_max_flux(L, i, maxd);
    capacity[frag_bulk] = 0;
}

void dflu_pack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluPack **pack) {
    DFluPack *p;
    int i, capacity[NBAGS];

    EMALLOC(1, pack);
    p = *pack;

    p->L = L;

    get_capacity(L, maxdensity, /**/ capacity);

    UC(dmap_ini(NFRAGS, capacity, /**/ &p->map));

    p->hii = p->hcc = NULL;
    p->dii = p->dcc = NULL;

    i = 0;
    p->hpp = &p->hbags[i];
    p->dpp = &p->dbags[i++];
    UC(comm_bags_ini(PINNED, NONE, sizeof(Particle), capacity, /**/ p->hpp, p->dpp));

    if (ids) {
        p->hii = &p->hbags[i];
        p->dii = &p->dbags[i++];
        UC(comm_bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ p->hii, p->dii));
    }

    if (colors) {
        p->hcc = &p->hbags[i];
        p->dcc = &p->dbags[i++];
        UC(comm_bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ p->hcc, p->dcc));
    }

    p->nbags = i;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void dflu_comm_ini(MPI_Comm cart, /**/ DFluComm **com) {
    DFluComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

static int nhalocells(int3 L) {
    return 8 +                  /* corners */
        4 * (L.x + L.y + L.z) + /* edges   */
        2 * L.x * L.y +         /* faces   */
        2 * L.x * L.z +
        2 * L.y * L.z;
}

void dflu_unpack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluUnpack **unpack) {
    int maxparts, i, capacity[NBAGS];
    DFluUnpack *u;

    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    maxparts = (int) (nhalocells(L) * maxdensity) + 1;

    get_capacity(L, maxdensity, /**/ capacity);

    u->hii = u->hcc = NULL;

    i = 0;
    u->hpp = &u->hbags[i++];
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, /**/ u->hpp, NULL));

    if (ids) {
        u->hii = &u->hbags[i++];
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ u->hii, NULL));
    }

    if (colors) {
        u->hcc = &u->hbags[i++];
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ u->hcc, NULL));
    }
    u->nbags = i;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));

    u->iire = u->ccre = NULL;
    Dalloc(&u->ppre, maxparts);
    if (ids)    Dalloc(&u->iire, maxparts);
    if (colors) Dalloc(&u->ccre, maxparts);
}
/* end src/distr/flu/imp/ini.h */
/* begin src/distr/flu/imp/fin.h */
void dflu_pack_fin(DFluPack *p) {
    UC(dmap_fin(NFRAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, NONE, /**/ p->hpp, p->dpp));

    if (p->hii) UC(comm_bags_fin(PINNED, NONE, /**/ p->hii, p->dii));
    if (p->hcc) UC(comm_bags_fin(PINNED, NONE, /**/ p->hcc, p->dcc));

    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void dflu_comm_fin(DFluComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void dflu_unpack_fin(DFluUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, u->hpp, NULL));

    if (u->hii) UC(comm_bags_fin(HST_ONLY, NONE, u->hii, NULL));
    if (u->hcc) UC(comm_bags_fin(HST_ONLY, NONE, u->hcc, NULL));

    UC(comm_buffer_fin(u->hbuf));

    Dfree(u->ppre);
    if (u->iire) Dfree(u->iire);
    if (u->ccre) Dfree(u->ccre);

    EFREE(u);
}
/* end src/distr/flu/imp/fin.h */
/* begin src/distr/flu/imp/map.h */
static void build_map(int3 L, int n, const PartList lp, DMap m) {
    UC(dmap_reini(NFRAGS, /**/ m));
    KL(dflu_dev::build_map, (k_cnf(n)), (L, lp, n, /**/ m));
    UC(dmap_download_counts(NFRAGS, /**/ &m)); /* async */
    KL(dmap_scan<NFRAGS>, (1, 32), (/**/ m));
}

void dflu_build_map(int n, const PartList lp, DFluPack *p) {
    UC(build_map(p->L, n, lp, p->map));
    dSync();
}
/* end src/distr/flu/imp/map.h */
/* begin src/distr/flu/imp/pack.h */
static int reduce(int n, const int d[]) {
    int s, i;
    for (i = s = 0; i < n; ++i) s += d[i];
    return s;
}

static void pack_pp(const DMap m, const Particle *pp, /**/ dBags *bags) {
    int n;
    const int S = sizeof(Particle) / sizeof(float2);
    float2p26 wrap;
    bag2Sarray(*bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dflu_dev::pack<float2, S>), (k_cnf(S*n)), ((const float2*)pp, m, /**/ wrap));
}

static void pack_ii(const DMap m, const int *ii, /**/ dBags *bags) {
    int n;
    const int S = 1;
    intp26 wrap;
    bag2Sarray(*bags, &wrap);
    n = reduce(NFRAGS, m.hcounts);

    KL((dflu_dev::pack<int, S>), (k_cnf(S*n)), (ii, m, /**/ wrap));
}

void dflu_pack(const FluQuants *q, /**/ DFluPack *p) {
    UC(pack_pp(p->map, q->pp, /**/ p->dpp));
    if (p->dii) UC(pack_ii(p->map, q->ii, /**/ p->dii));
    if (p->dcc) UC(pack_ii(p->map, q->cc, /**/ p->dcc));
}

struct ExceedData { int cap, cnt, fid; };
enum   {OK, FAIL};

static int check_counts(int nfrags, const int *counts, const hBags *hpp, /**/ ExceedData *e) {
    int fid, cnt, cap;
    for (fid = 0; fid < nfrags; ++fid) {
        cnt = counts[fid];
        cap = comm_get_number_capacity(fid, hpp);
        if (cnt > cap) {
            e->cap = cap; e->cnt = cnt; e->fid = fid;
            return FAIL;
        }
    }
    return OK;
}

static void fail_exceed(ExceedData *e) {
    enum {X, Y, Z};
    int cap, cnt, fid, d[3];
    cap = e->cap; cnt = e->cnt; fid = e->fid;
    frag_hst::i2d3(fid, d);
    ERR("exceed capacity, fragment %d = [%d %d %d]: %d/%d",
        fid, d[X], d[Y], d[Z], cnt, cap);
}

static void download_pp_and_counts(DFluPack *p) {
    size_t sz;
    int *cnt;
    sz = NFRAGS * sizeof(int);
    cnt = p->map.hcounts;
    dSync(); /* wait for pack kernels */
    memcpy(p->hpp->counts, cnt, sz);
    if (p->hii) memcpy(p->hii->counts, cnt, sz);
    if (p->hcc) memcpy(p->hcc->counts, cnt, sz);
    p->nhalo = reduce(NFRAGS, cnt);
}

void dflu_download(DFluPack *p, /**/ DFluStatus *s) {
    ExceedData e;
    int r;
    int *cnt;
    cnt = p->map.hcounts;
    r = check_counts(NFRAGS, cnt, p->hpp, /**/ &e);
    if (r == OK) {
        UC(download_pp_and_counts(p));
    }
    else {
        if   (dflu_status_nullp(s)) UC(fail_exceed(&e));
        else UC(dflu_status_exceed(e.fid, e.cnt, e.cap, /**/ s));
    }
}
/* end src/distr/flu/imp/pack.h */
/* begin src/distr/flu/imp/com.h */
void dflu_post_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void dflu_post_send(DFluPack *p, DFluComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void dflu_wait_recv(DFluComm *c, DFluUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void dflu_wait_send(DFluComm *c) {
    UC(comm_wait_send(c->comm));
}
/* end src/distr/flu/imp/com.h */
/* begin src/distr/flu/imp/unpack.h */
static int scan_hst(const int n, const int *counts, int27 *starts) {
    int i, s;
    starts->d[0] = 0;
    for (i = 0, s = 0; i < n; ++i)
        starts->d[i + 1] = (s += counts[i]);
    return s;
}

template <typename T>
static void unpack(const hBags *bags, int27 starts, /**/ T *buf) {
    int c, s, i;
    size_t sz, bs = bags->bsize;

    assert(bs == sizeof(T));

    for (i = 0; i < NFRAGS; ++i) {
        c = bags->counts[i];
        sz = c * bs;
        s = starts.d[i];
        if (c)
            CC(d::MemcpyAsync(buf + s, bags->data[i], sz, H2D));
    }
}

static int unpack_pp(int3 L, const hBags *bags, /**/ Particle *pp) {
    int nhalo;
    int27 starts;

    nhalo = scan_hst(NFRAGS, bags->counts, &starts);
    unpack(bags, starts, /**/ pp);

    dcommon_shift_halo(L, nhalo, starts, /**/ pp);

    return nhalo;
}

static int unpack_ii(const hBags *bags, /**/ int *ii) {
    int nhalo;
    int27 starts;

    nhalo = scan_hst(NFRAGS, bags->counts, &starts);
    unpack(bags, starts, /**/ ii);

    return nhalo;
}

static void unpack_pp(/**/ DFluUnpack *u) {
    u->nhalo = unpack_pp(u->L, u->hpp, /**/ u->ppre);
}

static void unpack_ii(/**/ DFluUnpack *u) {
    if (u->hii) unpack_ii(u->hii, /**/ u->iire);
}

static void unpack_cc(/**/ DFluUnpack *u) {
    if (u->hcc) unpack_ii(u->hcc, /**/ u->ccre);
}

void dflu_unpack(/**/ DFluUnpack *u) {
    unpack_pp(/**/ u);
    unpack_ii(/**/ u);
    unpack_cc(/**/ u);
}
/* end src/distr/flu/imp/unpack.h */
/* begin src/distr/flu/imp/gather.h */
void dflu_bulk(PartList lp, /**/ FluQuants *q) {
    UC(clist_ini_counts(q->cells));
    UC(clist_subindex_local(q->n, lp, /**/ q->cells, q->mcells));
}

void dflu_halo(const DFluUnpack *u, /**/ FluQuants *q) {
    PartList lp;
    lp.pp = u->ppre;
    lp.deathlist = NULL;
    UC(clist_subindex_remote(u->nhalo, lp, /**/ q->cells, q->mcells));
}

void dflu_gather(int ndead, const DFluPack *p, const DFluUnpack *u, /**/ FluQuants *q) {
    int n, nold, nhalo, nbulk;
    Particle *pp, *pp0;
    nold = q->n;
    nhalo = u->nhalo;
    nbulk = nold - p->nhalo - ndead;
    n = nbulk + nhalo;
    pp = q->pp; pp0 = q->pp0;

    const int nn[] = {nold, nhalo};
    UC(clist_build_map(nn, /**/ q->cells, q->mcells));

    UC(clist_gather_pp(pp, u->ppre, q->mcells, n, /**/ pp0));

    int *ii, *ii0, *cc, *cc0;
    ii = q->ii; ii0 = q->ii0;
    cc = q->cc; cc0 = q->cc0;

    if (p->hii) UC(clist_gather_ii(ii, u->iire, q->mcells, n, /**/ ii0));
    if (p->hcc) UC(clist_gather_ii(cc, u->ccre, q->mcells, n, /**/ cc0));

    q->n = n;

    /* swap pointers */

    q->pp = pp0;
    q->pp0 = pp;

    if (p->hii) {
        q->ii = ii0;
        q->ii0 = ii;
    }

    if (p->hcc) {
        q->cc = cc0;
        q->cc0 = cc;
    }
}
/* end src/distr/flu/imp/gather.h */
/* end src/distr/flu/imp.cu */

#line 1 "src/distr/flu/status/imp.cpp"

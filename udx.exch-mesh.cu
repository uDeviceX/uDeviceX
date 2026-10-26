// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/exch/mesh/imp.cu */
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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
/* begin src/inc/type.h */
#ifndef UDX_EXCH_TYPE_COMMON
#define UDX_EXCH_TYPE_COMMON
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
#endif /* UDX_EXCH_TYPE_COMMON */
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
#ifndef UDX_EXCH_FRAG_DEV_COMMON
#define UDX_EXCH_FRAG_DEV_COMMON
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
#endif /* UDX_EXCH_FRAG_DEV_COMMON */

/* begin src/d/api.h */
#ifndef UDX_EXCH_DEVICE_API_COMMON
#define UDX_EXCH_DEVICE_API_COMMON
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
#endif /* UDX_EXCH_DEVICE_API_COMMON */
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
#ifndef UDX_EXCH_KL_COMMON
#define UDX_EXCH_KL_COMMON
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
#endif /* UDX_EXCH_KL_COMMON */
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

/* begin src/algo/minmax/imp.h */
struct Particle;
struct float3;
// tag::interface[]
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);
// end::interface[]
/* end src/algo/minmax/imp.h */
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

/* begin src/frag/imp.h */
#ifndef UDX_EXCH_FRAG_IMP_COMMON
#define UDX_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_EXCH_FRAG_IMP_COMMON */
/* begin src/comm/imp.h */
#ifndef UDX_EXCH_COMM_COMMON
#define UDX_EXCH_COMM_COMMON
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
#endif /* UDX_EXCH_COMM_COMMON */
/* begin src/comm/utils.h */
#ifndef UDX_EXCH_COMM_UTILS_COMMON
#define UDX_EXCH_COMM_UTILS_COMMON
// tag::convert[]
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
// end::convert[]
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
/* end src/comm/utils.h */
#endif /* UDX_EXCH_COMM_UTILS_COMMON */

/* begin src/exch/map/type.h */
#ifndef UDX_EXCH_MAP_TYPE_COMMON
#define UDX_EXCH_MAP_TYPE_COMMON
// [E]xchange map

// tag::struct[]
enum {MAX_FRAGS=26};

struct EMap {
    int *counts;         /* number of entities leaving in each fragment */
    int *starts;         /* cumulative sum of the above                 */
    int *offsets;        /* offsets per fragment for each solute        */
    int *ids[MAX_FRAGS]; /* indices of leaving objects                  */
    int *cap;            /* capacity of ids                             */
};
// end::struct[]

/* end src/exch/map/type.h */
#endif /* UDX_EXCH_MAP_TYPE_COMMON */
/* begin src/exch/map/imp.h */
#ifndef UDX_EXCH_MAP_IMP_COMMON
#define UDX_EXCH_MAP_IMP_COMMON
// tag::int[]
void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map);               // <1>
void emap_fin(int nfrags, EMap *map);                                       // <2>
void emap_reini(int nw, int nfrags, /**/ EMap map);                         // <3>
void emap_scan(int nw, int nfrags, /**/ EMap map);                          // <4>
void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]); // <5>
void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int tot_counts[], int *counts[]);
// end::int[]
/* end src/exch/map/imp.h */
#endif /* UDX_EXCH_MAP_IMP_COMMON */

/* begin src/exch/mesh/imp.h */
struct EMeshPack;
struct EMeshComm;
struct EMeshUnpack;
struct EMeshPackM;
struct EMeshCommM;
struct EMeshUnpackM;

/* mesh exchanger */

// tag::mem[]
void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **p);
void emesh_comm_ini(MPI_Comm comm, /**/ EMeshComm **c);
void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **u);

void emesh_pack_fin(EMeshPack *p);
void emesh_comm_fin(EMeshComm *c);
void emesh_unpack_fin(EMeshUnpack *u);
// end::mem[]

// tag::map[]
void emesh_build_map(int nm, int nv, const Particle *pp, /**/ EMeshPack *p);
// end::map[]

// tag::pack[]
void emesh_pack_pp(int nv, const Particle *pp, /**/ EMeshPack *p);
void emesh_pack_rrcp(int nv, const Positioncp *rr, /**/ EMeshPack *p);
void emesh_download(EMeshPack *p);
// end::pack[]

// tag::com[]
void emesh_post_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_post_send(EMeshPack *p, EMeshComm *c);
void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_wait_send(EMeshComm *c);
// end::com[]

// tag::unpack[]
void emesh_unpack_pp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Particle *pp);
void emesh_unpack_rrcp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Positioncp *rr);
// end::unpack[]

// tag::get[]
void emesh_get_num_frag_mesh(const EMeshUnpack *u, /**/ int cc[NFRAGS]);
// end::get[]

/* optional: (back) momentum exchanger */

// tag::memback[]
void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshPackM **p);
void emesh_commm_ini(MPI_Comm comm, /**/ EMeshCommM **c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshUnpackM **u);

void emesh_packm_fin(EMeshPackM *p);
void emesh_commm_fin(EMeshCommM *c);
void emesh_unpackm_fin(EMeshUnpackM *u);
// end::memback[]

// tag::packback[]
void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm, /**/ EMeshPackM *p);
void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p);
// end::packback[]

// tag::comback[]
void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_post_send(EMeshPackM *p, EMeshCommM *c);
void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_wait_send(EMeshCommM *c);
// end::comback[]

// tag::unpackback[]
void emesh_upload(EMeshUnpackM *u);
void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u, /**/ Momentum *mm);
// end::unpackback[]
/* end src/exch/mesh/imp.h */

/* begin src/exch/map/dev.h */
#ifndef UDX_EXCH_MAP_DEV_COMMON
#define UDX_EXCH_MAP_DEV_COMMON
#define _I_ static __device__
#define _S_ static __device__

/* corner fragments are neighbor with 7 fragments */
enum { MAX_DSTS = 7 };

// tag::int[]
_I_ int emap_code(int3 L, const float r[3]) // <1>
// end::int[]
{
    int x, y, z;
    enum {X, Y, Z};
    x = -1 + (r[X] >= -L.x / 2) + (r[X] >= L.x / 2);
    y = -1 + (r[Y] >= -L.y / 2) + (r[Y] >= L.y / 2);
    z = -1 + (r[Z] >= -L.z / 2) + (r[Z] >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

// tag::int[]
_I_ int emap_code_box(int3 L, float3 lo, float3 hi)  // <2>
// end::int[]
{
    int x, y, z;
    x = -1 + (lo.x >= -L.x / 2) + (hi.x >= L.x / 2);
    y = -1 + (lo.y >= -L.y / 2) + (hi.y >= L.y / 2);
    z = -1 + (lo.z >= -L.z / 2) + (hi.z >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

// tag::int[]
_I_ void emap_add(int nfrags, int soluteid, int pid, int fid, EMap m) // <3>
// end::int[]
{
    int ientry, centry, stride, cap;
    cap = m.cap[fid];
    stride = nfrags + 1;
    centry = soluteid * stride + fid;
    ientry = atomicAdd(m.counts + centry, 1);
    if (ientry < cap)
        m.ids[fid][ientry] = pid;
}

_S_ int add_faces(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    for (int c = 0; c < 3; ++c) {
        if (d[c]) {
            int df[3] = {0, 0, 0}; df[c] = d[c];
            fids[j++] = frag_dev::d32i(df);
        }
    }
    return j;
}

_S_ int add_edges(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    for (int c = 0; c < 3; ++c) {
        int de[3] = {d[X], d[Y], d[Z]}; de[c] = 0;
        if (de[(c + 1) % 3] && de[(c + 2) % 3])
            fids[j++] = frag_dev::d32i(de);
    }
    return j;
}

_S_ int add_cornr(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    if (d[X] && d[Y] && d[Z])
        fids[j++] = frag_dev::d32i(d);
    return j;
}

// tag::int[]
_I_ int emap_decode(int code, /**/ int fids[MAX_DSTS]) // <4>
// end::int[]
{
    int j = 0;
    int d[3];
    frag_dev::i2d3(code, d);
    j = add_faces(j, d, /**/ fids);
    j = add_edges(j, d, /**/ fids);
    j = add_cornr(j, d, /**/ fids);
    return j;
}

#undef _I_
#undef _S_
/* end src/exch/map/dev.h */
#endif /* UDX_EXCH_MAP_DEV_COMMON */
/* begin src/exch/common/imp.h */
#ifndef UDX_EXCH_COMMON_IMP_COMMON
#define UDX_EXCH_COMMON_IMP_COMMON
struct int3;

// tag::struct[]
struct PackHelper {
    int *starts;
    int *offsets;
    int *indices[NFRAGS];
    int *cap;
};
// end::struct[]

// tag::int[]
void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf); // <1>
void ecommon_shift_pp_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void ecommon_shift_rrcp_one_frag(int3 L, int n, const int fid, /**/ Positioncp *rr);
// end::int[]
/* end src/exch/common/imp.h */
#endif /* UDX_EXCH_COMMON_IMP_COMMON */

/* begin src/exch/mesh/imp/type.h */
enum {
    EMESH_MAX_NBAGS = 1
};

struct EMeshPack {
    EMap map;
    float3 *minext, *maxext;
    dBags dbags[EMESH_MAX_NBAGS], *dpp;
    hBags hbags[EMESH_MAX_NBAGS], *hpp;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

struct EMeshComm {
    Comm *pp;
};

struct EMeshUnpack {
    hBags hbags[EMESH_MAX_NBAGS], *hpp;
    dBags dbags[EMESH_MAX_NBAGS], *dpp;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

/* optional structures for sending momenta back */

struct MMap { /* map for compression of Momentum (support structure only) */
    int *cc, *ss, *subids;
};

enum {
    ID_MM, ID_II,
    MAX_NMBAGS
};

struct EMeshPackM {
    MMap maps[NFRAGS];
    int *cchst, *ccdev; /* helper to collect counts */
    dBags dbags[MAX_NMBAGS], *dmm, *dii;
    hBags hbags[MAX_NMBAGS], *hmm, *hii;
    int nbags;
    CommBuffer *hbuf;
};

struct EMeshCommM {
    Comm *comm;
};

struct EMeshUnpackM {
    dBags dbags[MAX_NMBAGS], *dmm, *dii;
    hBags hbags[MAX_NMBAGS], *hmm, *hii;
    int nbags;
    CommBuffer *hbuf;
};

typedef Sarray<MMap, 26> MMap26;
/* end src/exch/mesh/imp/type.h */
/* begin src/exch/mesh/dev.h */
namespace emesh_dev {

/* L0: subdomain cropped by one cutoff radius */
__global__ void build_map(int3 L0, int n, const float3 *minext, const float3 *maxext, /**/ EMap map) {
    int i, fid, fids[MAX_DSTS], ndsts, j;
    float3 lo, hi;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    lo = minext[i];
    hi = maxext[i];

    fid = emap_code_box(L0, lo, hi);
    ndsts = emap_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        emap_add(NFRAGS, 0, i, fids[j], /**/ map);
}

template <typename T>
static __device__ void pack_p(int nv, const T *pp, int vid, int frag_mid, int *indices, /**/ T *buf) {
    int dst, src, mid;
    mid = __ldg(indices + frag_mid);
    dst = nv * frag_mid + vid;
    src = nv * mid      + vid;
    buf[dst] = pp[src];
}

template <typename T>
__global__ void pack_mesh(int nv, const T *pp, EMap map, /**/ Sarray<T*, 26> buf) {
    int gid, hi, step, fid, mid, vid, frag_mid;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = map.starts[26] * nv;
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        mid = gid / nv; /* mesh id   */
        vid = gid % nv; /* vertex id */
        fid = frag_dev::frag_get_fid(map.starts, mid);

        /* mesh index in the fragment coordinates */
        frag_mid = mid - map.starts[fid];

        if (frag_mid < map.cap[fid])
            pack_p(nv, pp, vid, frag_mid, map.ids[fid], /**/ buf.d[fid]);
    }
}

/* compress kernels */

enum {SKIP=-1};

// TODO this is copy paste from meshbb, might be removed from there
static __device__ bool nz(float a) {return fabs(a) > 1e-6f;}

static __device__ bool nonzero(const Momentum *m) {
    enum {X, Y, Z};
    return nz(m->P[X]) && nz(m->P[Y]) && nz(m->P[Z]) &&
        nz(m->L[X]) && nz(m->L[Y]) && nz(m->L[Z]);
}

__global__ void subindex_compress(int nt, int nm, const Momentum *mm, /**/ int *counts, int *subids) {
    int i, mid, subid;
    Momentum m;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nt * nm) return;

    mid = i / nt;

    m = mm[i];

    if (nonzero(&m))
        subid = atomicAdd(counts + mid, 1);
    else
        subid = SKIP;
    subids[i] = subid;
}

/* inclusive scan */
static __device__ int warpScan(int val) {
    int tid;
    tid = threadIdx.x % warpSize;
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * shfl_up(val, L) ;
    return val;
}

/* exclusive scan on one block */
template<int NWARP>
__global__ void block_scan(int n, const int *cc, /**/ int *ss) {
    assert(n < blockDim.x);
    assert(gridDim.x == 1);

    int i, tid, wid, lid, val, cnt, ws;
    tid = threadIdx.x;
    wid = tid / warpSize;
    lid = tid % warpSize;

    i = tid + blockIdx.x * blockDim.x;

    __shared__ int warp_cnt[NWARP];

    cnt = val = 0;

    if (i < n) cnt = cc[i];

    val = warpScan(cnt);

    if (lid == warpSize - 1) warp_cnt[wid] = val;

    __syncthreads();

    if (wid == 0) {
        int v, c;
        v = c = 0;
        if (lid < NWARP) c = warp_cnt[lid];
        v = warpScan(c);
        if (lid < NWARP) warp_cnt[lid] = v - c;
    }

    __syncthreads();

    ws = warp_cnt[wid];

    val += ws;

    if (i <= n) ss[i] = val - cnt;
}

__global__ void compress(int nt, int nm, const Momentum *mm, const int *starts, const int *subids, /**/ int *ids, Momentum *mmc) {
    int i, mid, subid, entry;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nt * nm) return;

    mid = i / nt;

    subid = subids[i];

    if (subid == SKIP) return;

    entry = subid + starts[mid];
    mmc[entry] = mm[i];
    ids[i] = i;
}

__global__ void collect_counts(int26 nm, MMap26 mm, /**/ int *counts) {
    int i, c, end;
    i = threadIdx.x;
    if (i > 26) return;
    end = nm.d[i];
    c = mm.d[i].ss[end];
    counts[i] = c;
}

static __device__ void addMom(const Momentum src, /**/ Momentum *dst) {
    enum {X, Y, Z};
    atomicAdd(dst->P + X, src.P[X]);
    atomicAdd(dst->P + Y, src.P[Y]);
    atomicAdd(dst->P + Z, src.P[Z]);

    atomicAdd(dst->L + X, src.L[X]);
    atomicAdd(dst->L + Y, src.L[Y]);
    atomicAdd(dst->L + Z, src.L[Z]);
}

static __device__ void unpack_mom_frag(int nt, int i, const int *hii, const Momentum *hmm, const int *mapii, /**/ Momentum *mm) {
    int hid, hmid, tid, lmid, lid;

    hid = hii[i];        /* global index in frag                */
    hmid = hid / nt;     /* mesh index in frag                  */
    tid  = hid % nt;     /* triangle id (same in frag or local) */

    lmid = mapii[hmid];  /* mesh index in bulk                  */

    lid = lmid * nt + tid;

    addMom(hmm[i], /**/ mm + lid);
}

__global__ void unpack_mom(int nt, int27 fragstarts, intp26 hii, Mop26 hmm, EMap map, /**/ Momentum *mm) {
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    fid = frag_dev::frag_get_fid(fragstarts.d, i);

    if (i >= fragstarts.d[26]) return;

    i -= fragstarts.d[fid];

    unpack_mom_frag(nt, i, hii.d[fid], hmm.d[fid], map.ids[fid], /**/ mm);
}

} // dev
/* end src/exch/mesh/dev.h */
/* begin src/exch/mesh/imp/ini.h */
// TODO this can be refined
static void get_capacity(int nfrags, int max_mesh_num, /**/ int cap[]) {
    for (int i = 0; i < nfrags; ++i)
        cap[i] = max_mesh_num;
}

void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **pack) {
    int i, cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    EMeshPack *p;

    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    UC(emap_ini(1, NFRAGS, cap, /**/ &p->map));

    i = 0;
    p->dpp = &p->dbags[i];
    p->hpp = &p->hbags[i++];
    UC(comm_bags_ini(PINNED, NONE, msz, cap, /**/ p->hpp, p->dpp));

    p->nbags = i;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));

    Dalloc(&p->minext, max_mesh_num);
    Dalloc(&p->maxext, max_mesh_num);
}

void emesh_comm_ini(MPI_Comm cart, /**/ EMeshComm **com) {
    EMeshComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->pp));
}

void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **unpack) {
    int i, cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    EMeshUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    i = 0;
    u->hpp = &u->hbags[i];
    u->dpp = &u->dbags[i++];
    UC(comm_bags_ini(PINNED_DEV, NONE, msz, cap, /**/ u->hpp, u->dpp));

    u->nbags = i;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}


/* Momentum structures */

static void get_mcap(int nfrags, int nt, const int *cap, /**/ int *mcap) {
    for (int i = 0; i < nfrags; ++i)
        mcap[i] = nt * cap[i];
}

static void ini_map(int nt, int max_mesh_num, MMap *map) {
    Dalloc(&map->cc, max_mesh_num);
    Dalloc(&map->ss, max_mesh_num);
    Dalloc(&map->subids, max_mesh_num * nt);
}

void emesh_packm_ini(int nt, int max_mesh_num, EMeshPackM **pack) {
    int i, cap[NFRAGS], mcap[NFRAGS];
    EMeshPackM *p;

    EMALLOC(1, pack);
    p = *pack;

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);

    for (i = 0; i < NFRAGS; ++i)
        UC(ini_map(nt, cap[i], /**/ &p->maps[i]));

    p->hmm = &p->hbags[ID_MM]; p->hii = &p->hbags[ID_II];
    p->dmm = &p->dbags[ID_MM]; p->dii = &p->dbags[ID_II];

    UC(comm_bags_ini(PINNED,   NONE, sizeof(Momentum), mcap, /**/ p->hmm, p->dmm));
    UC(comm_bags_ini(PINNED,   NONE, sizeof(int)     , mcap, /**/ p->hii, p->dii));

    p->nbags = MAX_NMBAGS;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));

    CC(d::alloc_pinned((void**) &p->cchst, 26 * sizeof(int)));
    CC(d::HostGetDevicePointer((void**) &p->ccdev, p->cchst, 0));
}

void emesh_commm_ini(MPI_Comm cart, /**/ EMeshCommM **com) {
    EMeshCommM *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void emesh_unpackm_ini(int nt, int max_mesh_num, EMeshUnpackM **unpack) {
    int cap[NFRAGS], mcap[NFRAGS];
    EMeshUnpackM *u;
    EMALLOC(1, unpack);
    u = *unpack;

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);

    u->hmm = &u->hbags[ID_MM]; u->hii = &u->hbags[ID_II];
    u->dmm = &u->dbags[ID_MM]; u->dii = &u->dbags[ID_II];

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Momentum), mcap, /**/ u->hmm, u->dmm));
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(int)     , mcap, /**/ u->hii, u->dii));

    u->nbags = MAX_NMBAGS;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}
/* end src/exch/mesh/imp/ini.h */
/* begin src/exch/mesh/imp/fin.h */
void emesh_pack_fin(EMeshPack *p) {
    UC(emap_fin(NFRAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, NONE, /**/ p->hpp, p->dpp));
    UC(comm_buffer_fin(p->hbuf));
    Dfree(p->minext);
    Dfree(p->maxext);
    EFREE(p);
}

void emesh_comm_fin(EMeshComm *c) {
    UC(comm_fin(c->pp));
    EFREE(c);
}

void emesh_unpack_fin(EMeshUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hpp, u->dpp));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}

/* Momentum struct */

static void fin_map(/**/ MMap *map) {
    Dfree(map->cc);
    Dfree(map->ss);
    Dfree(map->subids);
}

void emesh_packm_fin(EMeshPackM *p) {
    UC(comm_bags_fin(PINNED, NONE, /**/ p->hmm, p->dmm));
    UC(comm_bags_fin(PINNED, NONE, /**/ p->hii, p->dii));

    for (int i = 0; i < NFRAGS; ++i)
        UC(fin_map(&p->maps[i]));

    CC(d::FreeHost(p->cchst));
    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void emesh_commm_fin(EMeshCommM *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void emesh_unpackm_fin(EMeshUnpackM *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hmm, u->dmm));
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hii, u->dii));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}
/* end src/exch/mesh/imp/fin.h */
/* begin src/exch/mesh/imp/map.h */
static void fill_map(int3 L, int n, const float3 *lo, const float3 *hi, /**/ EMap map) {
    int3 L0 = make_int3(L.x-2, L.y-2, L.z-2);
    KL(emesh_dev::build_map, (k_cnf(n)), (L0, n, lo, hi, /**/ map));
}

void emesh_build_map(int nm, int nv, const Particle *pp, /**/ EMeshPack *p) {
    emap_reini(1, NFRAGS, /**/ p->map);
    minmax(pp, nv, nm, /**/ p->minext, p->maxext);
    fill_map(p->L, nm, p->minext, p->maxext, /**/ p->map);
    emap_scan(1, NFRAGS, /**/ p->map);
}
/* end src/exch/mesh/imp/map.h */
/* begin src/exch/mesh/imp/pack.h */
static void pack_mesh_pp(int nv, const Particle *pp, EMap map, /**/ Pap26 buf) {
    KL(emesh_dev::pack_mesh, (14 * 16, 128), (nv, pp, map, /**/ buf));
}

void emesh_pack_pp(int nv, const Particle *pp, /**/ EMeshPack *p) {
    Pap26 wrap;
    bag2Sarray(*p->dpp, &wrap);
    pack_mesh_pp(nv, pp, p->map, /**/ wrap);
}

static void pack_mesh_rrcp(int nv, const Positioncp *rr, EMap map, /**/ Pocpp26 buf) {
    KL(emesh_dev::pack_mesh, (14 * 16, 128), (nv, rr, map, /**/ buf));
}

void emesh_pack_rrcp(int nv, const Positioncp *rr, /**/ EMeshPack *p) {
    Pocpp26 wrap;
    bag2Sarray(*p->dpp, &wrap);
    pack_mesh_rrcp(nv, rr, p->map, /**/ wrap);
}

void emesh_download(EMeshPack *p) {
    int nw = 1;
    emap_download_tot_counts(nw, NFRAGS, p->map, /**/ p->hpp->counts);
}

static void reini_map(int nm, /**/ MMap *m) {
    if (nm) CC(d::MemsetAsync(m->cc, 0, nm * sizeof(int)));
}

static void compress_mom(int nt, int nm, const Momentum *mm, /**/ MMap *m, int *ids, Momentum *mmc) {
    UC(reini_map(nm, /**/ m));
    KL(emesh_dev::subindex_compress, (k_cnf(nt * nm)), (nt, nm, mm, /**/ m->cc, m->subids));

    enum {NWRP=4, WRPSZ=32};
    enum {BLCK=1, THRD=NWRP*WRPSZ};
    KL(emesh_dev::block_scan<NWRP>, (BLCK, THRD), (nm, m->cc, /**/ m->ss));

    KL(emesh_dev::compress, (k_cnf(nt * nm)), (nt, nm, mm, m->ss, m->subids, /**/ ids, mmc));
}

static void pack_mom(int nt, const int counts[NFRAGS], const Momentum *mm,
                     /**/ MMap maps[NFRAGS], int *ibuf[NFRAGS], Momentum *mbuf[NFRAGS]) {
    int i, s, c;

    for (i = s = 0; i < NFRAGS; ++i) {
        c = counts[i];
        UC(compress_mom(nt, c, mm + nt * s, /**/ maps + i, ibuf[i], mbuf[i]));
    }
}

void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm, /**/ EMeshPackM *p) {
    pack_mom(nt, counts, mm, /**/ p->maps, (int **) p->dii->data, (Momentum **) p->dmm->data);
}

void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p) {
    MMap26 mm;
    int26 ii;
    size_t sz;
    sz = NFRAGS * sizeof(int);

    memcpy(ii.d, counts, sz);
    memcpy(mm.d, p->maps, NFRAGS * sizeof(MMap));

    KL(emesh_dev::collect_counts, (32, 1), (ii, mm, /**/ p->ccdev));

    dSync();

    memcpy(p->hmm->counts, p->cchst, sz);
    memcpy(p->hii->counts, p->cchst, sz);
}

/* end src/exch/mesh/imp/pack.h */
/* begin src/exch/mesh/imp/com.h */
void emesh_post_recv(EMeshComm *c, EMeshUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->pp));
}

void emesh_post_send(EMeshPack *p, EMeshComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->pp));
}

void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u) {
    UC(comm_wait_recv(c->pp, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void emesh_wait_send(EMeshComm *c) {
    UC(comm_wait_send(c->pp));
}


/* momentum */

void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void emesh_post_send(EMeshPackM *p, EMeshCommM *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void emesh_wait_send(EMeshCommM *c) {
    UC(comm_wait_send(c->comm));
}
/* end src/exch/mesh/imp/com.h */
/* begin src/exch/mesh/imp/unpack.h */
static void upload_and_shift_pp(int n, int i, int3 L, const data_t *data, Particle *pp) {
    if (n == 0) return;
    size_t sz = n * sizeof(Particle);
    CC(d::MemcpyAsync(pp, data, sz, H2D));
    ecommon_shift_pp_one_frag(L, n, i, /**/ pp);
}

void emesh_unpack_pp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Particle *pp) {
    int i, nm, n, s = 0, nmtot = 0;

    for (i = 0; i < NFRAGS; ++i) {
        nm = u->hpp->counts[i];
        n  = nm * nv;
        upload_and_shift_pp(n, i, u->L, u->hpp->data[i], pp + s);
        s += n;
        nmtot += nm;
    }
    *nmhalo = nmtot;
}

static void upload_and_shift_rrcp(int n, int i, int3 L, const data_t *data, Positioncp *rr) {
    if (n == 0) return;
    size_t sz = n * sizeof(Positioncp);
    CC(d::MemcpyAsync(rr, data, sz, H2D));
    ecommon_shift_rrcp_one_frag(L, n, i, /**/ rr);
}

void emesh_unpack_rrcp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Positioncp *rr) {
    int i, nm, n, s = 0, nmtot = 0;

    for (i = 0; i < NFRAGS; ++i) {
        nm = u->hpp->counts[i];
        n  = nm * nv;
        upload_and_shift_rrcp(n, i, u->L, u->hpp->data[i], rr + s);
        s += n;
        nmtot += nm;
    }
    *nmhalo = nmtot;
}

void emesh_get_num_frag_mesh(const EMeshUnpack *u, /**/ int cc[NFRAGS]) {
    memcpy(cc, u->hpp->counts, NFRAGS * sizeof(int));
}

/* momentum */
static void upload_bags(const hBags *h, dBags *d) {
    int i, c;
    data_t *src, *dst;
    size_t sz;
    for (i = 0; i < NFRAGS; ++i) {
        c = h->counts[i];
        sz = c * h->bsize;
        src = h->data[i];
        dst = d->data[i];
        if (sz)
            CC(d::MemcpyAsync(dst, src, sz, H2D));
    }
}

void emesh_upload(EMeshUnpackM *u) {
    upload_bags(u->hmm, u->dmm);
    upload_bags(u->hii, u->dii);
}

static int get_fragstarts(int nfrags, const int cc[], /**/ int *starts) {
    int i, s;
    starts[0] = s = 0;
    for (i = 0; i < nfrags; ++i)
        starts[i+1] = (s += cc[i]);
    return s;
}

void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u, /**/ Momentum *mm) {
    intp26 wrapii;
    Mop26 wrapmm;
    int27 fragstarts;
    int n;

    bag2Sarray(*u->dmm, &wrapmm);
    bag2Sarray(*u->dii, &wrapii);

    n = get_fragstarts(NFRAGS, u->hii->counts, /**/ fragstarts.d);

    KL(emesh_dev::unpack_mom, (k_cnf(n)), (nt, fragstarts, wrapii, wrapmm, p->map, /**/ mm));
}
/* end src/exch/mesh/imp/unpack.h */
/* end src/exch/mesh/imp.cu */

#line 1 "src/exch/obj/imp.cu"

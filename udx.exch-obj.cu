// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/exch/obj/imp.cu */
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
/* begin src/exch/obj/imp.h */
struct EObjPack;
struct EObjComm;
struct EObjUnpack;

struct EObjPackF;
struct EObjCommF;
struct EObjUnpackF;

// tag::mem[]
void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **p);
void eobj_comm_ini(MPI_Comm cart, /**/ EObjComm **c);
void eobj_unpack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjUnpack **u);

void eobj_pack_fin(EObjPack *p);
void eobj_comm_fin(EObjComm *c);
void eobj_unpack_fin(EObjUnpack *u);
// end::mem[]

// tag::map[]
void eobj_build_map(int nw, const PaWrap *ww, /**/ EObjPack *p);
// end::map[]

// tag::pack[]
void eobj_pack(int nw, const PaWrap *ww, /**/ EObjPack *p);
void eobj_download(int nw, EObjPack *p);
// end::pack[]

// tag::com[]
void eobj_post_recv(EObjComm *c, EObjUnpack *u);
void eobj_post_send(EObjPack *p, EObjComm *c);
void eobj_wait_recv(EObjComm *c, EObjUnpack *u);
void eobj_wait_send(EObjComm *c);
// end::com[]

// tag::get[]
int26 eobj_get_counts(EObjUnpack *u); // <1>
void  eobj_get_all_counts(int nw, EObjUnpack *u, int *all_counts);
Pap26 eobj_upload_shift(EObjUnpack *u); // <2>
Fop26 eobj_reini_ff(const EObjUnpack *u, EObjPackF *pf); // <3>
// end::get[]


// tag::memback[]
void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **p);
void eobj_commf_ini(MPI_Comm cart, /**/ EObjCommF **c);
void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **u);

void eobj_packf_fin(EObjPackF *p);
void eobj_commf_fin(EObjCommF *c);
void eobj_unpackf_fin(EObjUnpackF *u);
// end::memback[]

// tag::packback[]
void eobj_download_ff(EObjPackF *p);
// end::packback[]

// tag::comback[]
void eobj_post_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_post_send_ff(EObjPackF *p, EObjCommF *c);
void eobj_wait_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_wait_send_ff(EObjCommF *c);
// end::comback[]

// tag::unpackback[]
void eobj_unpack_ff(EObjUnpackF *u, const EObjPack *p, int nw, /**/ FoWrap *ww);
// end::unpackback[]
/* end src/exch/obj/imp.h */

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

/* begin src/exch/obj/dev.h */
namespace eobj_dev {

__global__ void build_map(int3 L, int soluteid, int n, const Particle *pp, /**/ EMap map) {
    int pid, fid, fids[MAX_DSTS], ndsts, j;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    const Particle p = pp[pid];

    fid = emap_code(L, p.r);
    ndsts = emap_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        emap_add(NFRAGS, soluteid, pid, fids[j], /**/ map);
}

static __device__ void unpack_f(const Force *hff, int offset, int *indices, int i, /**/ Force *ff) {
    enum {X, Y, Z};
    int dst, src;

    src = offset + i;
    dst = __ldg(indices + src);

    const Force f = hff[src];

    atomicAdd(ff[dst].f + X, f.f[X]);
    atomicAdd(ff[dst].f + Y, f.f[Y]);
    atomicAdd(ff[dst].f + Z, f.f[Z]);
}

__global__ void unpack_ff(Fop26 hff, PackHelper ph, /**/ Force *ff) {
    int gid, fid, frag_i, hi, step;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = ph.starts[26];
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        fid = frag_dev::frag_get_fid(ph.starts, gid);

        /* index in the fragment coordinates */
        frag_i = gid - ph.starts[fid];

        unpack_f(hff.d[fid], ph.offsets[fid], ph.indices[fid], frag_i, /**/ ff);
    }
}

}
/* end src/exch/obj/dev.h */

/* begin src/exch/obj/imp/type.h */
/*
pp: particles
cc: counts of object types
*/

enum {
    ID_PP, ID_CC,
    EOBJ_MAX_NBAGS
};

struct EObjPack {
    EMap map;
    dBags dpp;
    hBags hbags[EOBJ_MAX_NBAGS], *hpp, *hcc;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

struct EObjComm {
    Comm *comm;
};

struct EObjUnpack {
    dBags dpp;
    hBags hbags[EOBJ_MAX_NBAGS], *hpp, *hcc;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

struct EObjPackF {
    dBags dff;
    hBags hff;
};

struct EObjCommF {
    Comm *comm;
};

struct EObjUnpackF {
    hBags hff;
    dBags dff;
};
/* end src/exch/obj/imp/type.h */
/* begin src/exch/obj/imp/ini.h */
static void estimates(int3 L, int nfrags, int maxd, int maxpsolid, int *cap) {
    int i, e, kind;
    frag_hst::estimates(L, nfrags, maxd, /**/ cap);

    /* estimates for solid */
    const float safety = 2.;
    float factor[3] = {safety / 2, safety / 4, safety / 8}; /* face, edge, corner */

    for (i = 0; i < nfrags; ++i) {
        int d[3];
        frag_hst::i2d3(i, d);
        kind = fabs(d[0]) + fabs(d[1]) + fabs(d[2]) - 1;
        e = maxpsolid * factor[kind];
        cap[i] += e;
    }
}

static void fill(int val, int n, int *a) {
    for (int i = 0; i < n; ++i) a[i] = val;
}

static void get_cc_cap(int nw, int nfrags, int *cap) {
    fill(nw, nfrags, cap);
}

static void set_cc_counts(int nw, int nfrags, hBags *ss) {
    fill(nw, nfrags, ss->counts);
}

void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **pack) {
    int cap[NFRAGS], cccap[NFRAGS];
    EObjPack * p;
    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);
    get_cc_cap(nw, NFRAGS, cccap);

    UC(emap_ini(nw, NFRAGS, cap, /**/ &p->map));

    p->hpp = &p->hbags[ID_PP];  p->hcc = &p->hbags[ID_CC];

    UC(comm_bags_ini(PINNED,   NONE, sizeof(Particle), cap, /**/ p->hpp, &p->dpp));
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int),    cccap, /**/ p->hcc, NULL));

    set_cc_counts(nw, NFRAGS, p->hcc);

    p->nbags = EOBJ_MAX_NBAGS;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void eobj_comm_ini(MPI_Comm cart, /**/ EObjComm **com) {
    EObjComm *c;
    EMALLOC(1, com);
    c = *com;

    UC(comm_ini(cart, /**/ &c->comm));
}

void eobj_unpack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjUnpack **unpack) {
    int cap[NFRAGS], cccap[NFRAGS];
    EObjUnpack *u;

    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);
    get_cc_cap(nw, NFRAGS, cccap);

    u->hpp = &u->hbags[ID_PP];  u->hcc = &u->hbags[ID_CC];

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ u->hpp, &u->dpp));
    UC(comm_bags_ini(HST_ONLY  , NONE, sizeof(int),    cccap, /**/ u->hcc, NULL));

    set_cc_counts(nw, NFRAGS, u->hcc);

    u->nbags = EOBJ_MAX_NBAGS;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}

void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **pack) {
    int cap[NFRAGS];
    EObjPackF *p;

    EMALLOC(1, pack);
    p = *pack;

    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &p->hff, &p->dff));
}

void eobj_commf_ini(MPI_Comm cart, /**/ EObjCommF **com) {
    EObjCommF *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **unpack) {
    int cap[NFRAGS];
    EObjUnpackF *u;

    EMALLOC(1, unpack);
    u = *unpack;

    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &u->hff, &u->dff));
}
/* end src/exch/obj/imp/ini.h */
/* begin src/exch/obj/imp/fin.h */
void eobj_pack_fin(EObjPack *p) {
    UC(emap_fin(NFRAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED,   NONE, /**/ p->hpp, &p->dpp));
    UC(comm_bags_fin(HST_ONLY, NONE, /**/ p->hcc, NULL));
    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void eobj_comm_fin(EObjComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void eobj_unpack_fin(EObjUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hpp, &u->dpp));
    UC(comm_bags_fin(HST_ONLY  , NONE, /**/ u->hcc, NULL));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}

void eobj_packf_fin(EObjPackF *p) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &p->hff, &p->dff));
    EFREE(p);
}

void eobj_commf_fin(EObjCommF *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void eobj_unpackf_fin(EObjUnpackF *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hff, &u->dff));
    EFREE(u);
}
/* end src/exch/obj/imp/fin.h */
/* begin src/exch/obj/imp/map.h */
static void add_wrap_to_map(int3 L, int wid, int n, const Particle *pp, EMap map) {
    int3 L0 = make_int3(L.x-2, L.y-2, L.z-2);
    KL(eobj_dev::build_map, (k_cnf(n)), (L0, wid, n, pp, /**/ map));
}

static void fill_map(int3 L, int nw, const PaWrap *ww, /**/ EMap map) {
    const PaWrap *w;
    for (int i = 0; i < nw; ++i) {
        w = ww + i;
        add_wrap_to_map(L, i, w->n, w->pp, /**/ map);
    }
}

static void build_map(int3 L, int nw, const PaWrap *ww, /**/ EMap map) {
    emap_reini(nw, NFRAGS, /**/ map);
    fill_map(L, nw, ww, /**/ map);
    emap_scan(nw, NFRAGS, /**/ map);
}

void eobj_build_map(int nw, const PaWrap *ww, /**/ EObjPack *p) {
    if (!nw) return;
    build_map(p->L, nw, ww, /**/ p->map);
}
/* end src/exch/obj/imp/map.h */
/* begin src/exch/obj/imp/pack.h */
static void pack_pp(int nfrags, int nw, const PaWrap *ww, EMap map, /**/ Pap26 buf) {
    int i, stride;
    stride = nfrags + 1;
    const PaWrap *w;
    PackHelper ph;

    for (i = 0; i < nw; ++i) {
        w = ww + i;
        ph.starts  = map.starts  + i * stride;
        ph.offsets = map.offsets + i * stride;
        ph.cap = map.cap;
        memcpy(ph.indices, map.ids, nfrags * sizeof(int*));

        UC(ecommon_pack_pp(w->pp, ph, /**/ buf));
    }
}

void eobj_pack(int nw, const PaWrap *ww, /**/ EObjPack *p) {
    Pap26 wrap;
    UC(bag2Sarray(p->dpp, &wrap));
    UC(pack_pp(NFRAGS, nw, ww, p->map, /**/ wrap));
}

void eobj_download(int nw, EObjPack *p) {
    if (!nw) return;
    UC(emap_download_all_counts(nw, NFRAGS, p->map, /**/ p->hpp->counts, (int**) p->hcc->data));
}

static void clear_forces(int nfrags, /**/ EObjPackF *p) {
    int i, c;
    size_t sz;
    for (i = 0; i < nfrags; ++i) {
        c = p->hff.counts[i];
        sz = c * p->hff.bsize;
        if (c) CC(d::MemsetAsync(p->dff.data[i], 0, sz));
    }
}

Fop26 eobj_reini_ff(const EObjUnpack *u, EObjPackF *pf) {
    size_t sz = NBAGS * sizeof(int);
    memcpy(pf->hff.counts, u->hpp->counts, sz);
    clear_forces(NFRAGS, /**/ pf);

    Fop26 wrap;
    bag2Sarray(pf->dff, &wrap);
    return wrap;
}

void eobj_download_ff(EObjPackF *p) {
    int i, c;
    size_t sz;
    data_t *src, *dst;
    for (i = 0; i < NFRAGS; ++i) {
        c = p->hff.counts[i];
        sz = c * p->hff.bsize;
        src = p->dff.data[i];
        dst = p->hff.data[i];
        if (c) CC(d::MemcpyAsync(dst, src, sz, D2H));
    }
    dSync();
}
/* end src/exch/obj/imp/pack.h */
/* begin src/exch/obj/imp/com.h */
void eobj_post_recv(EObjComm *c, EObjUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void eobj_post_send(EObjPack *p, EObjComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void eobj_wait_recv(EObjComm *c, EObjUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void eobj_wait_send(EObjComm *c) {
    UC(comm_wait_send(c->comm));
}

void eobj_post_recv_ff(EObjCommF *c, EObjUnpackF *u) {
    UC(comm_post_recv(&u->hff, c->comm));
}

void eobj_post_send_ff(EObjPackF *p, EObjCommF *c) {
    UC(comm_post_send(&p->hff, c->comm));
}

void eobj_wait_recv_ff(EObjCommF *c, EObjUnpackF *u) {
    UC(comm_wait_recv(c->comm, &u->hff));
}

void eobj_wait_send_ff(EObjCommF *c) {
    UC(comm_wait_send(c->comm));
}
/* end src/exch/obj/imp/com.h */
/* begin src/exch/obj/imp/unpack.h */
int26 eobj_get_counts(EObjUnpack *u) {
    int26 cc;
    memcpy(cc.d, u->hpp->counts, NFRAGS * sizeof(int));
    return cc;
}

void  eobj_get_all_counts(int nw, EObjUnpack *u, int *all_counts) {
    hBags *cc = u->hcc;
    int i, j, *ci;
    for (i = 0; i < NFRAGS; ++i) {
        ci = (int*) cc->data[i];
        for (j = 0; j < nw; ++j)
            all_counts[j * NFRAGS + i] = ci[j];
    }
}

static void upload(int nfrags, const hBags *h, /**/ dBags *d) {
    int i, c;
    size_t sz;
    data_t *src, *dst;

    for (i = 0; i < nfrags; ++i) {
        c = h->counts[i];
        if (c) {
            sz  = h->bsize * c;
            dst = d->data[i];
            src = h->data[i];
            CC(d::MemcpyAsync(dst, src, sz, H2D));
        }
    }
}

static void shift_pp(int3 L, int nfrags, const int counts[], /**/ dBags d) {
    int i, n;
    Particle *pp;

    for (i = 0; i < nfrags; ++i) {
        n = counts[i];
        if (n) {
            pp = (Particle *) d.data[i];
            ecommon_shift_pp_one_frag(L, n, i, /**/ pp);
        }
    }
}

Pap26 eobj_upload_shift(EObjUnpack *u) {
    upload(NFRAGS, u->hpp, /**/ &u->dpp);
    shift_pp(u->L, NFRAGS, u->hpp->counts, /**/ u->dpp);
    Pap26 wrap;
    bag2Sarray(u->dpp, &wrap);
    return wrap;
}

static void unpack_ff(int nfrags, Fop26 ff, EMap map, int nw, /**/ FoWrap *ww) {
    int i, stride;
    stride = nfrags + 1;
    const FoWrap *w;
    PackHelper ph;

    for (i = 0; i < nw; ++i) {
        w = ww + i;
        ph.starts  = map.starts  + i * stride;
        ph.offsets = map.offsets + i * stride;
        memcpy(ph.indices, map.ids, nfrags * sizeof(int*));

        KL(eobj_dev::unpack_ff, (14 * 16, 128), (ff, ph, /**/ w->ff));
    }
}

void eobj_unpack_ff(EObjUnpackF *u, const EObjPack *p, int nw, /**/ FoWrap *ww) {
    Fop26 wrap;
    upload(NFRAGS, &u->hff, /**/ &u->dff);
    bag2Sarray(u->dff, &wrap);
    unpack_ff(NFRAGS, wrap, p->map, nw, /**/ ww);
}
/* end src/exch/obj/imp/unpack.h */
/* end src/exch/obj/imp.cu */

#line 1 "src/flu/imp.cpp"

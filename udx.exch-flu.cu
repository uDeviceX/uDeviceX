// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/exch/flu/imp.cu */
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
/* begin src/frag/dev.h */
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
/* begin src/frag/imp.h */
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

/* begin src/comm/imp.h */
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
/* begin src/comm/utils.h */
// tag::convert[]
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
// end::convert[]
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
/* end src/comm/utils.h */

/* begin src/struct/parray/imp.h */
struct PaArray {
    bool colors;
    const float *pp;
    const int *cc;
};

struct PaArray_v;
struct PaCArray_v;

struct Particle;

// tag::push[]
void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);
// end::push[]

// tag::get[]
bool parray_is_colored(const PaArray *a);
// end::get[]

// tag::view[]
void parray_get_view(const PaArray *a, PaArray_v *v);  // <1>
void parray_get_view(const PaArray *a, PaCArray_v *v); // <2>
// end::view[]
/* end src/struct/parray/imp.h */
/* begin src/flu/type.h */
/* helpers to communicate between halo interactions and exchanger */

namespace flu {

enum { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct LFrag { /* "local" fragment */
    PaArray parray;
    const int *ii; /* index */
    int n;
};

struct RFrag { /* "remote" fragment */
    PaArray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

struct RndFrag {
    float seed;
    int mask;
};

typedef Sarray<LFrag,   26> LFrag26;
typedef Sarray<RFrag,   26> RFrag26;
typedef Sarray<RndFrag, 26> RndFrag26;

} // flu
/* end src/flu/type.h */
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

/* begin src/exch/flu/imp.h */
struct EFluPack;
struct EFluComm;
struct EFluUnpack;
struct PaArray;

// tag::mem[]
void eflu_pack_ini(bool colors, int3 L, int maxd, EFluPack **p);
void eflu_comm_ini(MPI_Comm comm, /**/ EFluComm **c);
void eflu_unpack_ini(bool colors, int3 L, int maxd, EFluUnpack **u);

void eflu_pack_fin(EFluPack *p);
void eflu_comm_fin(EFluComm *c);
void eflu_unpack_fin(EFluUnpack *u);
// end::mem[]

// tag::map[]
void eflu_compute_map(const int *start, const int *count, /**/ EFluPack *p);
void eflu_download_cell_starts(/**/ EFluPack *p);
// end::map[]

// tag::pack[]
void eflu_pack(const PaArray *parray, /**/ EFluPack *p);
void eflu_download_data(EFluPack *p);
// end::pack[]

// tag::com[]
void eflu_post_recv(EFluComm *c, EFluUnpack *u);
void eflu_post_send(EFluPack *p, EFluComm *c);
void eflu_wait_recv(EFluComm *c, EFluUnpack *u);
void eflu_wait_send(EFluComm *c);
// tag::end[]

// tag::unpack[]
void eflu_unpack(EFluUnpack *u);

using flu::LFrag26;
using flu::RFrag26;

void eflu_get_local_frags(const EFluPack *p, /**/ LFrag26 *lfrags);
void eflu_get_remote_frags(const EFluUnpack *u, /**/ RFrag26 *rfrags);
// end::unpack[]

/* end src/exch/flu/imp.h */

namespace eflu_dev {
/* begin src/exch/flu/dev/map.h */
/* return origin and extents of the cells of fragment i */
static __device__ void get_frag_box(int3 L0, int i, /**/ int org[3], int ext[3]) {
    int d[3];
    int c, L[3] = {L0.x, L0.y, L0.z};
    frag_dev::i2d3(i, d);
    for (c = 0; c < 3; ++c) {
        org[c] = (d[c] == 1) ? L[c] - 1 : 0;
        ext[c] = (d[c] == 0) ? L[c]     : 1;
    }
}

/* convert cellid from frag coordinates to bulk coordinates */
static __device__ int frag2bulk(int3 L, int hci, const int org[3], const int ext[3]) {
    enum {X, Y, Z};
    int c;
    int src[3];
    int dst[3] = {hci % ext[X], (hci / ext[X]) % ext[Y], hci / (ext[X] * ext[Y])};
    for (c = 0; c < 3; ++c) src[c] = org[c] + dst[c];
    return src[X] + L.x * (src[Y] + L.y * src[Z]);
}

/* collect cell informations from bulk cells to frag cells */
__global__ void count_cells(int3 L, const int27 cellpackstarts, const int *start, const int *count,
                            /**/ intp26 fragss, intp26 fragcc) {
    enum {X, Y, Z};
    int gid, fid; /* fragment id */
    int nhc; /* number of halo cells */
    int cid, hci; /* bulk and halo cell ids */
    int org[3], ext[3]; /* fragment origin and extend */
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= cellpackstarts.d[26]) return;

    fid = frag_dev::frag_get_fid(cellpackstarts.d, gid);
    hci = gid - cellpackstarts.d[fid];

    get_frag_box(L, fid, /**/ org, ext);
    nhc = ext[X] * ext[Y] * ext[Z];

    if (hci < nhc) {
        cid = frag2bulk(L, hci, org, ext);
        fragss.d[fid][hci] = start[cid];
        fragcc.d[fid][hci] = count[cid];
    } else if (hci == nhc) {
        fragss.d[fid][hci] = 0;
        fragcc.d[fid][hci] = 0;
    }
}

/* convert cell starts from bulk coordinates to fragment coordinates */
template <int NWARPS>
__global__ void scan(const int26 fragn, const intp26 fragcc, /**/ intp26 fragcum) {
    __shared__ int shdata[32];

    int tid, laneid, warpid, fid;
    int *count, *start, n;
    int lastval, sourcebase, sourceid, mycount, myscan;
    int L, val, gs;

    fid = blockIdx.x;
    count = fragcc.d[fid];
    start = fragcum.d[fid];
    n = fragn.d[fid];

    tid = threadIdx.x;
    laneid = threadIdx.x % warpSize;
    warpid = threadIdx.x / warpSize;

    lastval = 0;
    for (sourcebase = 0; sourcebase < n; sourcebase += 32 * NWARPS) {
        sourceid = sourcebase + tid;
        mycount = myscan = 0;
        if (sourceid < n) myscan = mycount = count[sourceid];
        if (tid == 0) myscan += lastval;

        for (L = 1; L < 32; L <<= 1) {
            val = shfl_up(myscan, L);
            if (laneid >= L) myscan += val;
        }

        if (laneid == 31) shdata[warpid] = myscan;
        __syncthreads();
        if (warpid == 0) {
            gs = 0;
            if (laneid < NWARPS) gs = shdata[tid];
            for ( L = 1; L < 32; L <<= 1) {
                val = shfl_up(gs, L);
                if (laneid >= L) gs += val;
            }

            shdata[tid] = gs;
            lastval = shfl(gs, 31);
        }
        __syncthreads();
        if (warpid) myscan += shdata[warpid - 1];
        __syncthreads();
        if (sourceid < n) start[sourceid] = myscan - mycount;
    }
}
/* end src/exch/flu/dev/map.h */
/* begin src/exch/flu/dev/pack.h */
/* copy particles from bulk array pp to a single fragment to be exchanged */
__global__ void collect_particles(int fid, int nc, const Particle *pp, const int *bss,
                                  const int *bcc, const int *fss, int cap,
                                  /**/ int *fii, Particle *fpp, int *fnn) {
    int cid, tid, src, dst, nsrc, nfloat2s;
    int i, lpid, dpid, spid, c;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* cid: work group id = cell id      */
    /* tid: worker id within the group   */


    cid = threadIdx.x / (warpSize / 2) + 2 * blockIdx.x;
    tid = threadIdx.x % (warpSize / 2);

    if (cid >= nc) return;

    src = bss[cid];
    dst = fss[cid];
    nsrc = min(bcc[cid], cap - dst);

    const float2 *srcpp = (const float2*) pp;
    float2 *dstpp = (float2*) fpp;

    nfloat2s = nsrc * 3;
    for (i = tid; i < nfloat2s; i += warpSize/2) {
        lpid = i / 3;
        c    = i % 3;
        dpid = dst + lpid;
        spid = src + lpid;
        dstpp[3 * dpid + c] = srcpp[3 * spid + c];
    }
    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fii[dpid] = spid;
    }
    if (cid + 1 == nc) fnn[fid] = dst;
}

/* copy colors from bulk array ii to a single fragment to be exchanged */
__global__ void collect_colors(int fid, int nc, const int *ii,
                               const int *bss, const int *bcc, const int *fss,
                               int cap, /**/ int *fii) {
    int cid, tid, src, dst, nsrc;
    int lpid, dpid, spid;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* cid: work group id = cell id      */
    /* tid: worker id within the group   */

    cid = threadIdx.x / (warpSize / 2) + 2 * blockIdx.x;
    tid = threadIdx.x % (warpSize / 2);

    if (cid >= nc) return;

    src = bss[cid];
    dst = fss[cid];
    nsrc = min(bcc[cid], cap - dst);

    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fii[dpid] = ii[spid];
    }
}

/* end src/exch/flu/dev/pack.h */
} // dev

/* begin src/exch/flu/imp/type.h */
/*
  pp: exchanged particles
  cc: colors of the above

  bcc: (bulk counts): cell counts
  bss: (bulk starts): cell starts in bulk coordinates
  fss: (fragment starts): cell starts in fragment coordinates

  bii: indices of the particles in bulk coordinates
*/

struct EFluPack {
    enum {
        MAX_NHBAGS = 3,
        MAX_NDBAGS = 2
    };

    intp26 bcc, bss, bii, fss;
    int *counts_dev;
    int26 cap;

    dBags dbags[MAX_NDBAGS], *dpp, *dcc;
    hBags hbags[MAX_NHBAGS], *hpp, *hcc, *hfss;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

struct EFluComm {
    Comm *comm;
};

struct EFluUnpack {
    enum {
        MAX_NHBAGS = 3,
        MAX_NDBAGS = 3
    };

    dBags dbags[MAX_NDBAGS], *dpp, *dcc, *dfss;
    hBags hbags[MAX_NHBAGS], *hpp, *hcc, *hfss;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};
/* end src/exch/flu/imp/type.h */
/* begin src/exch/flu/imp/ini.h */
void eflu_pack_ini(bool colors, int3 L, int maxd, EFluPack **pack) {
    int i, nbags, nc, cap[NBAGS], ncs[NBAGS];
    EFluPack *p;

    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    frag_hst::estimates(L, NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    memcpy(p->cap.d, cap, NFRAGS * sizeof(int));

    for (i = 0; i < NFRAGS; ++i) {
        ncs[i] = nc = frag_hst::ncell(L, i) + 1;
        Dalloc(&p->bcc.d[i], nc);
        Dalloc(&p->bss.d[i], nc);
        Dalloc(&p->fss.d[i], nc);

        Dalloc(&p->bii.d[i], cap[i]);
    }
    ncs[BULK] = 0;

    nbags = 0;
    p->hpp = &p->hbags[nbags];
    p->dpp = &p->dbags[nbags++];
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ p->hpp, p->dpp));

    p->hcc = NULL; p->dcc = NULL;
    if (colors) {
        p->hcc = &p->hbags[nbags];
        p->dcc = &p->dbags[nbags++];
        UC(comm_bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ p->hcc, p->dcc));
    }

    p->hfss = &p->hbags[nbags++];
    UC(comm_bags_ini(PINNED_HST, NONE, sizeof(int), ncs, /**/ p->hfss, NULL));
    p->nbags = nbags;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));

    memcpy(p->hfss->counts, ncs, sizeof(ncs));

    Dalloc(&p->counts_dev, NFRAGS);
}

void eflu_comm_ini(MPI_Comm cart, /**/ EFluComm **com) {
    EFluComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void eflu_unpack_ini(bool colors, int3 L, int maxd, EFluUnpack **unpack) {
    int nbags, i, cap[NBAGS], ncs[NBAGS];
    EFluUnpack *u;

    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    frag_hst::estimates(L, NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;

    for (i = 0; i < NFRAGS; ++i)
        ncs[i] = frag_hst::ncell(L, i) + 1;
    ncs[BULK] = 0;

    nbags = 0;

    u->hpp = &u->hbags[nbags];
    u->dpp = &u->dbags[nbags++];
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ u->hpp, u->dpp));

    u->hcc = NULL; u->dcc = NULL;
    if (colors) {
        u->hcc = &u->hbags[nbags];
        u->dcc = &u->dbags[nbags++];
        UC(comm_bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ u->hcc, u->dcc));
    }

    u->hfss = &u->hbags[nbags];
    u->dfss = &u->dbags[nbags++];
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(int), ncs, /**/ u->hfss, u->dfss));

    u->nbags = nbags;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}

/* end src/exch/flu/imp/ini.h */
/* begin src/exch/flu/imp/fin.h */
void eflu_pack_fin(EFluPack *p) {
    for (int i = 0; i < NFRAGS; ++i) {
        Dfree(p->bcc.d[i]);
        Dfree(p->bss.d[i]);
        Dfree(p->fss.d[i]);
        Dfree(p->bii.d[i]);
    }

    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ p->hpp, p->dpp));
    if (p->hcc)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ p->hcc, p->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ p->hfss, NULL));
    UC(comm_buffer_fin(p->hbuf));

    Dfree(p->counts_dev);
    EFREE(p);
}

void eflu_comm_fin(EFluComm *c) {
    UC(comm_fin(/**/ c->comm));
    EFREE(c);
}

void eflu_unpack_fin(EFluUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hpp, u->dpp));
    if (u->hcc)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hcc, u->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ u->hfss, u->dfss));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}

/* end src/exch/flu/imp/fin.h */
/* begin src/exch/flu/imp/common.h */
static int get_cell_num(int3 L, int *cc) {
    int i, nc, c;
    for (nc = i = 0; i < NFRAGS; ++i) {
        c = cc[i] = frag_hst::ncell(L, i) + 1;
        nc += c;
    }
    return nc;
}

static void scan(int n, int *cc, /**/ int *ss) {
    ss[0] = 0;
    for (int i = 0; i < n; ++i) ss[i+1] = ss[i] + cc[i];
}
/* end src/exch/flu/imp/common.h */
/* begin src/exch/flu/imp/map.h */
void eflu_compute_map(const int *start, const int *count, /**/ EFluPack *p) {
    int nc;
    int26 cc;
    int27 ss;
    nc = get_cell_num(p->L, /**/ cc.d);
    scan(NFRAGS, cc.d, /**/ ss.d);
    KL(eflu_dev::count_cells, (k_cnf(nc)), (p->L, ss, start, count, /**/ p->bss, p->bcc));
    KL(eflu_dev::scan<32>, (26, 32 * 32), (cc, p->bcc, /**/ p->fss));
}


static void download_cell_starts(int3 L, intp26 src, /**/ intp26 dst) {
    int i, nc;
    size_t sz;

    for(i = 0; i < NFRAGS; ++i) {
        nc = frag_hst::ncell(L, i) + 1;
        sz = nc * sizeof(int);
        CC(d::MemcpyAsync(dst.d[i], src.d[i], sz, D2H));
    }
}

void eflu_download_cell_starts(/**/ EFluPack *p) {
    intp26 fss_hst;
    bag2Sarray(*p->hfss, /**/ &fss_hst);
    download_cell_starts(p->L, p->fss, /**/ fss_hst);
    /* size of the messages is fixed throughout the whole simulation */
    /* sizes are frag_ncell(fid) + 1 */
}

/* end src/exch/flu/imp/map.h */
/* begin src/exch/flu/imp/pack.h */
void eflu_pack(const PaArray *parray, /**/ EFluPack *p) {
    int nc, i, cap;
    int26 cc;
    Pap26 fpp;
    intp26 fcc;
    const int *bss, *bcc, *fss;

    CC(d::MemsetAsync(p->counts_dev, 0, NFRAGS * sizeof(int)));

    get_cell_num(p->L, /**/ cc.d);

    bag2Sarray(*p->dpp, /**/ &fpp);
    if (parray->colors) bag2Sarray(*p->dcc, /**/ &fcc);

    for (i = 0; i < 26; ++i) {
        nc = cc.d[i];
        bss = p->bss.d[i];
        bcc = p->bcc.d[i];
        fss = p->fss.d[i];
        cap = p->cap.d[i];

        KL( eflu_dev::collect_particles,
            ((nc+1) / 2, 32),
            (i, nc, (const Particle*) parray->pp, bss, bcc, fss, cap, /**/ p->bii.d[i], fpp.d[i], p->counts_dev));

        if (parray->colors)
            KL( eflu_dev::collect_colors,
                ((nc+1) / 2, 32),
                (i, nc, parray->cc, bss, bcc, fss, cap, /**/ fcc.d[i]));
    }
}

static void copy(int, const int counts[], const dBags *d, /**/ hBags *h) {
    int i, c;
    size_t sz;

    for (i = 0; i < NFRAGS; ++i) {
        c = counts[i];
        sz = c * h->bsize;
        if (c)
            CC(d::MemcpyAsync(h->data[i], d->data[i], sz, D2H));
    }
}

void eflu_download_data(EFluPack *p) {
    int counts[NFRAGS];
    size_t sz = sizeof(counts);

    CC(d::MemcpyAsync(counts, p->counts_dev, sz, D2H));
    dSync(); /* wait for counts memcpy */

    copy(NFRAGS, counts, p->dpp, /**/ p->hpp);
    if (p->hcc)
        copy(NFRAGS, counts, p->dcc, /**/ p->hcc);

    memcpy(p->hpp->counts, counts, sz);
    if (p->hcc)
        memcpy(p->hcc->counts, counts, sz);

    dSync(); /* wait for copy */
}
/* end src/exch/flu/imp/pack.h */
/* begin src/exch/flu/imp/com.h */
void eflu_post_recv(EFluComm *c, EFluUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void eflu_post_send(EFluPack *p, EFluComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void eflu_wait_recv(EFluComm *c, EFluUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void eflu_wait_send(EFluComm *c) {
    UC(comm_wait_send(c->comm));
}

/* end src/exch/flu/imp/com.h */
/* begin src/exch/flu/imp/unpack.h */
static void upload_data(int n, const hBags *h, /**/ dBags *d) {
    int i, c;
    size_t sz;
    for (i = 0; i < n; ++i) {
        c = h->counts[i];
        sz = c * h->bsize;
        if (c)
            CC(d::MemcpyAsync(d->data[i], h->data[i], sz, H2D));
    }
}

/* upload recved data on the device */
void eflu_unpack(EFluUnpack *u) {
    upload_data(NFRAGS, u->hfss, /**/ u->dfss);
    upload_data(NFRAGS, u->hpp, /**/ u->dpp);
    if (u->hcc)
        upload_data(NFRAGS, u->hcc, /**/ u->dcc);
}
/* end src/exch/flu/imp/unpack.h */
/* begin src/exch/flu/imp/get.h */
void eflu_get_local_frags(const EFluPack *p, /**/ LFrag26 *lfrags) {
    PaArray parraya;

    for (int i = 0; i < 26; ++i) {
        parray_push_pp((Particle*) p->dpp->data[i], &parraya);
        if (p->hcc)
            parray_push_cc((int*) p->dcc->data[i], &parraya);

        lfrags->d[i] = {
            .parray  = parraya,
            .ii = p->bii.d[i],
            .n  = p->hpp->counts[i]
        };
    }
}

void eflu_get_remote_frags(const EFluUnpack *u, /**/ RFrag26 *rfrags) {
    int i, dx, dy, dz, xcells, ycells, zcells;
    int3 L;
    PaArray parrayb;
    L = u->L;

    for (i = 0; i < 26; ++i) {
        dx = frag_hst::i2dx(i);
        dy = frag_hst::i2dy(i);
        dz = frag_hst::i2dz(i);

        xcells = dx == 0 ? L.x : 1;
        ycells = dy == 0 ? L.y : 1;
        zcells = dz == 0 ? L.z : 1;
        parray_push_pp((Particle*) u->dpp->data[i], &parrayb);
        if (u->hcc)
            parray_push_cc((int*) u->dcc->data[i], &parrayb);

        rfrags->d[i] = {
            .parray = parrayb,
            .start = (int*) u->dfss->data[i],
            .dx = dx,
            .dy = dy,
            .dz = dz,
            .xcells = xcells,
            .ycells = ycells,
            .zcells = zcells,
            .type = abs(dx) + abs(dy) + abs(dz)
        };
    }
}
/* end src/exch/flu/imp/get.h */
/* end src/exch/flu/imp.cu */

#line 1 "src/exch/map/imp.cu"

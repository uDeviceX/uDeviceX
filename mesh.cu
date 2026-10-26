
// Editable compilation unit for mesh exchange and mesh/structure coupling.


// Grouped editable compilation unit for mesh and object exchange.


#include "config.cuh"

#include <mpi.h>
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
#endif /* UDX_EXCH_TYPE_COMMON */
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
#endif /* UDX_EXCH_FRAG_DEV_COMMON */

#ifndef UDX_EXCH_DEVICE_API_COMMON
#define UDX_EXCH_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif /* UDX_EXCH_DEVICE_API_COMMON */
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
#ifndef UDX_EXCH_KL_COMMON
#define UDX_EXCH_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif /* UDX_EXCH_KL_COMMON */
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
struct float3;
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);
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

#ifndef UDX_EXCH_FRAG_IMP_COMMON
#define UDX_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_EXCH_FRAG_IMP_COMMON */
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

#endif /* UDX_EXCH_COMM_COMMON */
#ifndef UDX_EXCH_COMM_UTILS_COMMON
#define UDX_EXCH_COMM_UTILS_COMMON
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
#endif /* UDX_EXCH_COMM_UTILS_COMMON */

#ifndef UDX_EXCH_MAP_TYPE_COMMON
#define UDX_EXCH_MAP_TYPE_COMMON
// [E]xchange map

enum {MAX_FRAGS=26};

struct EMap {
    int *counts;         /* number of entities leaving in each fragment */
    int *starts;         /* cumulative sum of the above                 */
    int *offsets;        /* offsets per fragment for each solute        */
    int *ids[MAX_FRAGS]; /* indices of leaving objects                  */
    int *cap;            /* capacity of ids                             */
};

#endif /* UDX_EXCH_MAP_TYPE_COMMON */
#ifndef UDX_EXCH_MAP_IMP_COMMON
#define UDX_EXCH_MAP_IMP_COMMON
void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map);               // <1>
void emap_fin(int nfrags, EMap *map);                                       // <2>
void emap_reini(int nw, int nfrags, /**/ EMap map);                         // <3>
void emap_scan(int nw, int nfrags, /**/ EMap map);                          // <4>
void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]); // <5>
void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int tot_counts[], int *counts[]);
#endif /* UDX_EXCH_MAP_IMP_COMMON */

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

#ifndef UDX_EXCH_MAP_DEV_COMMON
#define UDX_EXCH_MAP_DEV_COMMON
#define _I_ static __device__
#define _S_ static __device__

/* corner fragments are neighbor with 7 fragments */
enum { MAX_DSTS = 7 };

_I_ int emap_code(int3 L, const float r[3]) // <1>
{
    int x, y, z;
    enum {X, Y, Z};
    x = -1 + (r[X] >= -L.x / 2) + (r[X] >= L.x / 2);
    y = -1 + (r[Y] >= -L.y / 2) + (r[Y] >= L.y / 2);
    z = -1 + (r[Z] >= -L.z / 2) + (r[Z] >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

_I_ int emap_code_box(int3 L, float3 lo, float3 hi)  // <2>
{
    int x, y, z;
    x = -1 + (lo.x >= -L.x / 2) + (hi.x >= L.x / 2);
    y = -1 + (lo.y >= -L.y / 2) + (hi.y >= L.y / 2);
    z = -1 + (lo.z >= -L.z / 2) + (hi.z >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

_I_ void emap_add(int nfrags, int soluteid, int pid, int fid, EMap m) // <3>
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

_I_ int emap_decode(int code, /**/ int fids[MAX_DSTS]) // <4>
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
#endif /* UDX_EXCH_MAP_DEV_COMMON */
#ifndef UDX_EXCH_COMMON_IMP_COMMON
#define UDX_EXCH_COMMON_IMP_COMMON
struct int3;

struct PackHelper {
    int *starts;
    int *offsets;
    int *indices[NFRAGS];
    int *cap;
};

void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf); // <1>
void ecommon_shift_pp_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void ecommon_shift_rrcp_one_frag(int3 L, int n, const int fid, /**/ Positioncp *rr);
#endif /* UDX_EXCH_COMMON_IMP_COMMON */

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





#include <mpi.h>
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
#endif /* UDX_EXCH_TYPE_COMMON */
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
#endif /* UDX_EXCH_FRAG_DEV_COMMON */
#ifndef UDX_EXCH_FRAG_IMP_COMMON
#define UDX_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_EXCH_FRAG_IMP_COMMON */

#ifndef UDX_EXCH_DEVICE_API_COMMON
#define UDX_EXCH_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif /* UDX_EXCH_DEVICE_API_COMMON */
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
#ifndef UDX_EXCH_KL_COMMON
#define UDX_EXCH_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif /* UDX_EXCH_KL_COMMON */
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

#endif /* UDX_EXCH_COMM_COMMON */
#ifndef UDX_EXCH_COMM_UTILS_COMMON
#define UDX_EXCH_COMM_UTILS_COMMON
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
#endif /* UDX_EXCH_COMM_UTILS_COMMON */

#ifndef UDX_EXCH_MAP_TYPE_COMMON
#define UDX_EXCH_MAP_TYPE_COMMON
// [E]xchange map

enum {MAX_FRAGS=26};

struct EMap {
    int *counts;         /* number of entities leaving in each fragment */
    int *starts;         /* cumulative sum of the above                 */
    int *offsets;        /* offsets per fragment for each solute        */
    int *ids[MAX_FRAGS]; /* indices of leaving objects                  */
    int *cap;            /* capacity of ids                             */
};

#endif /* UDX_EXCH_MAP_TYPE_COMMON */
#ifndef UDX_EXCH_MAP_IMP_COMMON
#define UDX_EXCH_MAP_IMP_COMMON
void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map);               // <1>
void emap_fin(int nfrags, EMap *map);                                       // <2>
void emap_reini(int nw, int nfrags, /**/ EMap map);                         // <3>
void emap_scan(int nw, int nfrags, /**/ EMap map);                          // <4>
void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]); // <5>
void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int tot_counts[], int *counts[]);
#endif /* UDX_EXCH_MAP_IMP_COMMON */
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

#ifndef UDX_EXCH_MAP_DEV_COMMON
#define UDX_EXCH_MAP_DEV_COMMON
#define _I_ static __device__
#define _S_ static __device__

/* corner fragments are neighbor with 7 fragments */
enum { MAX_DSTS = 7 };

_I_ int emap_code(int3 L, const float r[3]) // <1>
{
    int x, y, z;
    enum {X, Y, Z};
    x = -1 + (r[X] >= -L.x / 2) + (r[X] >= L.x / 2);
    y = -1 + (r[Y] >= -L.y / 2) + (r[Y] >= L.y / 2);
    z = -1 + (r[Z] >= -L.z / 2) + (r[Z] >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

_I_ int emap_code_box(int3 L, float3 lo, float3 hi)  // <2>
{
    int x, y, z;
    x = -1 + (lo.x >= -L.x / 2) + (hi.x >= L.x / 2);
    y = -1 + (lo.y >= -L.y / 2) + (hi.y >= L.y / 2);
    z = -1 + (lo.z >= -L.z / 2) + (hi.z >= L.z / 2);

    return frag_dev::d2i(x, y, z);
}

_I_ void emap_add(int nfrags, int soluteid, int pid, int fid, EMap m) // <3>
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

_I_ int emap_decode(int code, /**/ int fids[MAX_DSTS]) // <4>
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
#endif /* UDX_EXCH_MAP_DEV_COMMON */
#ifndef UDX_EXCH_COMMON_IMP_COMMON
#define UDX_EXCH_COMMON_IMP_COMMON
struct int3;

struct PackHelper {
    int *starts;
    int *offsets;
    int *indices[NFRAGS];
    int *cap;
};

void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf); // <1>
void ecommon_shift_pp_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void ecommon_shift_rrcp_one_frag(int3 L, int n, const int fid, /**/ Positioncp *rr);
#endif /* UDX_EXCH_COMMON_IMP_COMMON */

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





// Grouped editable compilation unit for mesh and structure subsystems.


// Grouped editable compilation unit for the complete mesh subsystem.


// Grouped editable compilation unit for mesh geometry, topology, coordinates, and generation.


// Grouped editable compilation unit for mesh geometry, measures, topology, and collision.


// Grouped editable compilation unit for mesh geometry and measures.


// Grouped editable compilation unit for mesh geometry helpers.



#include <stdio.h>
#include <math.h>
#include <vector_types.h>

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

namespace tri_hst {
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
}

struct MeshAngle;
struct Vectors;
struct MeshRead;

void mesh_angle_ini(MeshRead*, /**/ MeshAngle**); // <1>
void mesh_angle_fin(MeshAngle*);
void  mesh_angle_apply(MeshAngle*, int nm, Vectors*, /**/ double*); // <2>
struct MeshAngle {
    int nd, nv;
    int4 *dd; /* dihidrals */
};
void mesh_angle_ini(MeshRead *mesh, MeshAngle **pq) {
    int i, nv, nd;
    MeshAngle *q;
    const int4 *dd;
    UC(nv = mesh_read_get_nv(mesh));
    UC(nd = mesh_read_get_ne(mesh));
    UC(dd = mesh_read_get_dih(mesh));
    EMALLOC(1, &q);
    EMALLOC(nd, &q->dd);
    for (i = 0; i < nd; i++)
        q->dd[i] = dd[i];
    q->nd = nd; q->nv = nv;
    *pq = q;
}

void mesh_angle_fin(MeshAngle *q) {
    EFREE(q->dd); EFREE(q);
}

static void get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double angle(int4 t, Vectors *p, int offset) {
    int ia, ib, ic, id;
    double a[3], b[3], c[3], d[3], x, y;
    ia = t.x; ib = t.y; ic = t.z; id = t.w;

    UC(get(p, ia + offset, /**/ a));
    UC(get(p, ib + offset, /**/ b));
    UC(get(p, ic + offset, /**/ c));
    UC(get(p, id + offset, /**/ d));

    tri_hst::dihedral_xy(a, b, c, d, /**/ &x, &y);
    return atan2(y, x);
}
void mesh_angle_apply(MeshAngle *q, int m, Vectors *p, /**/ double *angle0) {
    int i, j, k, nv, nd, offset;
    int4 *dd;
    nd = q->nd; nv = q->nv; dd = q->dd; offset = 0;

    for (k = i = 0; i < m; i++) {
        for (j = 0; j < nd; j++)
            angle0[k++] = angle(dd[j], p, offset);
        offset += nv;
    }
}






#include <vector_types.h>

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
#endif
struct Particle;
struct float3;
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);

struct float3;
struct Particle;

void mesh_get_bbox(const float *rr, const int n, /**/ float3 *minbb, float3 *maxbb);

void mesh_get_bboxes_hst(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb);
void mesh_get_bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb);

static void T2r(const int i, const Particle *pp, const float **r) {*r = pp[i].r;}
static void T2r(const int i, const float *rr,    const float **r) {*r = rr + 3*i;}

static float min2(float a, float b) {return a < b ? a : b;}
static float max2(float a, float b) {return a < b ? b : a;}

template <typename T>
static void get_bbox_(const T *vv, const int n, /**/ float3 *minbb, float3 *maxbb) {
    enum {X, Y, Z};
    const float *r;
    float3 minb, maxb;

    if (n == 0) return;

    T2r(0, vv, /**/ &r);

    minb.x = r[X]; minb.y = r[Y]; minb.z = r[Z];
    maxb = minb;

    for (int i = 1; i < n; ++i) {
        T2r(i, vv, /**/ &r);
        minb.x = min2(minb.x, r[0]); maxb.x = max2(maxb.x, r[0]);
        minb.y = min2(minb.y, r[1]); maxb.y = max2(maxb.y, r[1]);
        minb.z = min2(minb.z, r[2]); maxb.z = max2(maxb.z, r[2]);
    }
    *minbb = minb; *maxbb = maxb;
}

void mesh_get_bbox(const float *rr, const int n, /**/ float3 *minbb, float3 *maxbb) {
    get_bbox_(rr, n, /**/ minbb, maxbb);
}

void mesh_get_bboxes_hst(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb) {
    for (int i = 0; i < ns; ++i)
    get_bbox_(pp + i*nps, nps, /**/ minbb + i, maxbb + i);
}

void mesh_get_bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb) {
    if (ns == 0) return;
    minmax(pp, nps, ns, /**/ minbb, maxbb);
}





// Grouped editable compilation unit for mesh area and length measures.


// Grouped editable compilation unit for mesh area calculations.



#include <stdio.h>
#include <vector_types.h>

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
struct KahanSum;

void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>

namespace tri_hst {
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
}

struct MeshArea;
struct Vectors;
struct MeshRead;

void mesh_area_ini(MeshRead*, /**/ MeshArea**); // <1>
void mesh_area_fin(MeshArea*);

double mesh_area_apply0(MeshArea*, Vectors*);                     // <2>
void  mesh_area_apply(MeshArea*, int nm, Vectors*, /**/ double*); // <3>
struct MeshArea {
    int4 *tt;
    int nv, nt;
};
void mesh_area_ini(MeshRead *mesh, MeshArea **pq) {
    int nv, nt;
    MeshArea *q;
    EMALLOC(1, &q);
    nt = mesh_read_get_nt(mesh);
    nv = mesh_read_get_nv(mesh);
    EMALLOC(nt, &q->tt);

    q->nv = nv; q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_area_fin(MeshArea *q) { EFREE(q->tt); EFREE(q); }
static void mesh_area_get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double mesh_area_value(int nt, int4 *tt, Vectors *p, int offset) {
    int i, ia, ib, ic;
    double a[3], b[3], c[3], sum;
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    for (i = 0; i < nt; i++) {
        ia = tt[i].x; ib = tt[i].y; ic = tt[i].z;
        UC(mesh_area_get(p, ia + offset, /**/ a));
        UC(mesh_area_get(p, ib + offset, /**/ b));
        UC(mesh_area_get(p, ic + offset, /**/ c));
        kahan_sum_add(kahan_sum, tri_hst::kahan_area(a, b, c));
    }
    sum = kahan_sum_get(kahan_sum);
    return sum;
}
double mesh_area_apply0(MeshArea *q, Vectors *p) {
    int nt, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; offset = 0;
    return mesh_area_value(nt, tt, p, offset);
}

void mesh_area_apply(MeshArea *q, int m, Vectors *p, double *area0) {
    int i, nt, nv, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; nv = q->nv; offset = 0;

    for (i = 0; i < m; i++) {
        UC(area0[i] = mesh_area_value(nt, tt, p, offset));
        offset += nv;
    }
}






#include <stdio.h>
#include <vector_types.h>

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

namespace tri_hst {
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
}

struct MeshTriArea;
struct Vectors;
struct MeshRead;

void mesh_tri_area_ini(MeshRead*, /**/ MeshTriArea**); // <1>
void mesh_tri_area_fin(MeshTriArea*);

void  mesh_tri_area_apply(MeshTriArea*, int nm, Vectors*, /**/ double*); // <2>
struct MeshTriArea {
    int4 *tt;
    int nv, nt;
};
void mesh_tri_area_ini(MeshRead *mesh, MeshTriArea **pq) {
    int nv, nt;
    MeshTriArea *q;
    EMALLOC(1, &q);
    nt = mesh_read_get_nt(mesh);
    nv = mesh_read_get_nv(mesh);
    EMALLOC(nt, &q->tt);

    q->nv = nv; q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_tri_area_fin(MeshTriArea *q) { EFREE(q->tt); EFREE(q); }
static void mesh_tri_area_get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double mesh_tri_area_value(int4 t, Vectors *p, int offset) {
    int ia, ib, ic;
    double a[3], b[3], c[3];
    ia = t.x; ib = t.y; ic = t.z;
    UC(mesh_tri_area_get(p, ia + offset, /**/ a));
    UC(mesh_tri_area_get(p, ib + offset, /**/ b));
    UC(mesh_tri_area_get(p, ic + offset, /**/ c));
    return tri_hst::kahan_area(a, b, c);
}

void mesh_tri_area_apply(MeshTriArea *q, int m, Vectors *p, double *area0) {
    int i, j, k, nt, nv, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; nv = q->nv; offset = 0;

    for (k = i = 0; i < m; i++) {
        for (j = 0; j < nt; j++)
            area0[k++] = mesh_tri_area_value(tt[j], p, offset);
        offset += nv;
    }
}






// Grouped editable compilation unit for edge lengths and vertex areas.



#include <stdio.h>
#include <math.h>
#include <vector_types.h>

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

struct MeshEdgLen;
struct MeshRead;
struct Vectors;

void mesh_edg_len_ini(MeshRead*, MeshEdgLen**); // <1>
void mesh_edg_len_fin(MeshEdgLen*);
void mesh_edg_len_apply(MeshEdgLen*, int nm, Vectors*, /**/ double*); // <2>
struct MeshEdgLen {
    int ne, nv;
    int *ee;
};
void mesh_edg_len_ini(MeshRead *mesh, MeshEdgLen **pq) {
    MeshEdgLen *q;
    int i, j, nv, ne, *ee;
    const int4 *dd;
    int4 d;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    dd = mesh_read_get_dih(mesh);
    EMALLOC(2*ne, &ee);
    for (j = i = 0; i < ne; i++) {
        d = dd[i];
        ee[j++] = d.y; ee[j++] = d.z;
    }
    q->nv = nv; q->ne = ne; q->ee = ee;
    *pq = q;
}

static void mesh_edg_len_get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double sum_sq(double x, double y, double z) { return x*x + y*y + z*z; }
static double len0(Vectors *pos, int i, int j) {
    enum {X, Y, Z};
    double d, a[3], b[3];
    mesh_edg_len_get(pos, i, /**/ a);
    mesh_edg_len_get(pos, j, /**/ b);
    d = sum_sq(a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]);
    return sqrt(d);
}
static void len(MeshEdgLen *q, Vectors *pos, int offset_v, int offset_e, /**/ double *o) {
    int ne, i, k, a, b;
    double value;
    const int *ee;
    ne = q->ne;
    ee = q->ee;
    for (k = i = 0; i < ne; i++) {
        a = ee[k++]; b = ee[k++];
        UC(value = len0(pos, a + offset_v, b + offset_v));
        o[i + offset_e] = value;
    }
}
void mesh_edg_len_apply(MeshEdgLen *q, int nm, Vectors *pos, /**/ double *o) {
    int i, nv, ne;
    int offset_v, offset_e; /* vert and edge */
    nv = q->nv;
    ne = q->ne;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(len(q, pos, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
}

void mesh_edg_len_fin(MeshEdgLen *q) {
    EFREE(q->ee);
    EFREE(q);
}





#include <stdio.h>
#include <vector_types.h>

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

namespace tri_hst {
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
}

struct MeshVertArea;
struct Vectors;
struct MeshRead;

void mesh_vert_area_ini(MeshRead*, /**/ MeshVertArea**); // <1>
void mesh_vert_area_fin(MeshVertArea*);

void  mesh_vert_area_apply(MeshVertArea*, int nm, Vectors*, /**/ double*); // <2>
struct MeshVertArea {
    int4 *tt;
    int nv, nt;
};
void mesh_vert_area_ini(MeshRead *mesh, MeshVertArea **pq) {
    int nv, nt;
    MeshVertArea *q;
    EMALLOC(1, &q);
    nt = mesh_read_get_nt(mesh);
    nv = mesh_read_get_nv(mesh);
    EMALLOC(nt, &q->tt);

    q->nv = nv; q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_vert_area_fin(MeshVertArea *q) { EFREE(q->tt); EFREE(q); }
static void mesh_vert_area_get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static void area(int4 t, Vectors *p, int offset, /**/ double *o) {
    int ia, ib, ic;
    double a[3], b[3], c[3], A;
    ia = t.x; ib = t.y; ic = t.z;
    UC(mesh_vert_area_get(p, ia + offset, /**/ a));
    UC(mesh_vert_area_get(p, ib + offset, /**/ b));
    UC(mesh_vert_area_get(p, ic + offset, /**/ c));

    A = tri_hst::kahan_area(a, b, c)/3;
    o[ia + offset] += A;
    o[ib + offset] += A;
    o[ic + offset] += A;
}

void mesh_vert_area_apply(MeshVertArea *q, int nm, Vectors *pos, double *o) {
    int i, j, nt, nv, n, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; nv = q->nv; offset = 0;
    n = nv * nm;
    for (i = 0; i < n; i++) o[i] = 0;

    for (i = 0; i < nm; i++) {
        for (j = 0; j < nt; j++)
            area(tt[j], pos, offset, /**/ o);
        offset += nv;
    }
}







// Grouped editable compilation unit for mesh topology and collision.


// Grouped editable compilation unit for mesh topology and volume helpers.



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


#ifndef UDX_EXCH_DEVICE_API_COMMON
#define UDX_EXCH_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
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

#ifndef UDX_MESH_TRIANGLES_TYPE
#define UDX_MESH_TRIANGLES_TYPE
struct Triangles {
    int4 *tt;
    int nt;
};
#endif
struct Triangles;
struct MeshRead;

void triangles_ini(MeshRead*, /**/ Triangles**);
void triangles_fin(Triangles*);

void triangles_ini(MeshRead *mesh, /**/ Triangles **pq) {
    int nt;
    Triangles *q;
    const int4 *hst;
    int4 *dev;
    EMALLOC(1, &q);
    *pq = q;
    nt = mesh_read_get_nt(mesh);
    hst = mesh_read_get_tri(mesh);
    Dalloc(&dev, nt);
    cH2D(dev, hst, nt);

    q->nt = nt; q->tt = dev;
}

void triangles_fin(Triangles *q) {
    Dfree(q->tt);
    EFREE(q);
}





#include <stdio.h>
#include <vector_types.h>

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
struct KahanSum;

void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>

struct MeshVolume;
struct Vectors;
struct MeshRead;

void mesh_volume_ini(MeshRead*, /**/ MeshVolume**); // <1>
void mesh_volume_fin(MeshVolume*);

double mesh_volume_apply0(MeshVolume*, Vectors*);                      // <2>
void  mesh_volume_apply(MeshVolume*, int nm, Vectors*, /**/ double*); // <3>
struct MeshVolume {
    double *rr; /* workspace for coordinates shifted to a mesh center
                   of mass */
    int4 *tt;
    int nv, nt;
};
void mesh_volume_ini(MeshRead *mesh, MeshVolume **pq) {
    int nv, nt;
    MeshVolume *q;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    EMALLOC(3*nv, &q->rr);
    EMALLOC(  nt, &q->tt);

    q->nv = nv;
    q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_volume_fin(MeshVolume *q) {
    EFREE(q->rr);
    EFREE(q->tt);
    EFREE(q);
}

static void compute_com(int nv, Vectors *pos, /**/ double *com) {
    enum {X, Y, Z};
    int i;
    float r[3];
    KahanSum *sx, *sy, *sz;
    kahan_sum_ini(&sx); kahan_sum_ini(&sy); kahan_sum_ini(&sz);
    for (i = 0; i < nv; i++) {
        UC(vectors_get(pos, i, /**/ r));
        kahan_sum_add(sx, r[X]);
        kahan_sum_add(sy, r[Y]);
        kahan_sum_add(sz, r[Z]);
    }
    com[X] = kahan_sum_get(sx)/nv;
    com[Y] = kahan_sum_get(sy)/nv;
    com[Z] = kahan_sum_get(sz)/nv;
    kahan_sum_fin(sx); kahan_sum_fin(sy); kahan_sum_fin(sz);
}
static void to_com(int nv, int offset, Vectors *pos, /**/ double *rr) {
    enum {X, Y, Z};
    int i;
    float   r0[3]; /* from */
    double *r1;    /* to */
    double com[3];
    UC(compute_com(nv, pos, /**/ com));
    for (i = 0; i < nv; i++) {
        UC(vectors_get(pos, i + offset, /**/ r0));
        r1 = &rr[3*i];
        r1[X] = r0[X] - com[X];
        r1[Y] = r0[Y] - com[Y];
        r1[Z] = r0[Z] - com[Z];
    }
}
static double M0(double a[3], double b[3], double c[3]) {
    enum {X, Y, Z};
    return  (+a[X]*(b[Y]*c[Z]-b[Z]*c[Y])
             -a[Y]*(b[X]*c[Z]-b[Z]*c[X])
             +a[Z]*(b[X]*c[Y]-b[Y]*c[X]));
}
static void get(double *rr, int i, /**/ double *r1) {
    enum {X, Y, Z};
    double *r0;
    r0 = &rr[3*i];
    r1[X] = r0[X]; r1[Y] = r0[Y]; r1[Z] = r0[Z];
}
static double volume(int nt, int4 *tt, double *rr) {
    enum {X, Y, Z};
    int i, ia, ib, ic;
    double a[3], b[3], c[3], sum;
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    for (i = 0; i < nt; i++) {
        ia = tt[i].x; ib = tt[i].y; ic = tt[i].z;
        UC(get(rr, ia, /**/ a));
        UC(get(rr, ib, /**/ b));
        UC(get(rr, ic, /**/ c));
        kahan_sum_add(kahan_sum, M0(a, b, c));
    }
    sum = kahan_sum_get(kahan_sum);
    return sum/6;
}
double mesh_volume_apply0(MeshVolume *q, Vectors *positions) {
    int nv, nt, offset;
    double *rr;
    int4 *tt;
    nv = q->nv; nt = q->nt; rr = q->rr; tt = q->tt; offset = 0;
    UC(to_com(nv, offset, positions, /**/ rr));
    return volume(nt, tt, rr);
}

 void mesh_volume_apply(MeshVolume *q, int m, Vectors *positions, double *volume0) {
    int i, nt, nv, offset;
    double *rr;
    int4 *tt;
    nv = q->nv; nt = q->nt; rr = q->rr; tt = q->tt; offset = 0;
    for (i = 0; i < m; i++) {
        UC(to_com(nv, offset, positions, /**/ rr));
        UC(volume0[i] = volume(nt, tt, rr));
        offset += nv;
    }
}






#include <stdio.h>
#include <vector_types.h>

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

struct MeshScatter;
struct MeshRead;
struct Scalars;

void mesh_scatter_ini(MeshRead*, MeshScatter**); // <1>
void mesh_scatter_fin(MeshScatter*);
void mesh_scatter_edg2vert(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
void mesh_scatter_edg2vert_avg(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
struct MeshScatter {
    int ne, nv;
    int *ee;
    int *deg; /* vert. degree */
};
static void deg_ini(int nv, int ne, const int *ee, /**/ int *deg) {
    int i, k, a, b;
    for (i = 0; i < nv; i++) deg[i] = 0;
    for (i = k = 0; i < ne; i++) {
        a = ee[k++];
        b = ee[k++];
        deg[a]++;
        deg[b]++;
    }
}
void mesh_scatter_ini(MeshRead *mesh, MeshScatter **pq) {
    MeshScatter *q;
    int i, j, nv, ne;
    int *ee, *deg;
    const int4 *dd;
    int4 d;
    EMALLOC(1, &q);

    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    dd = mesh_read_get_dih(mesh);
    EMALLOC(2*ne, &ee);
    for (j = i = 0; i < ne; i++) {
        d = dd[i];
        ee[j++] = d.y; ee[j++] = d.z;
    }
    EMALLOC(  nv, &deg);
    UC(deg_ini(nv, ne, ee, /**/ deg));

    q->nv = nv; q->ne = ne;
    q->ee = ee; q->deg = deg;
    *pq = q;
}

static void edg2vert(MeshScatter *q, Scalars *sc, int offset_v, int offset_e, /**/ double *o) {
    int ne, i, k, a, b;
    double value;
    const int *ee;
    ne = q->ne;
    ee = q->ee;
    for (k = i = 0; i < ne; i++) {
        a = ee[k++]; b = ee[k++];
        UC(value = scalars_get(sc, i + offset_e));
        o[a + offset_v] += value;
        o[b + offset_v] += value;
    }
}
void mesh_scatter_edg2vert_avg(MeshScatter *q, int nm, Scalars *sc, /**/ double *o) {
    int i, j, k, d, n, nv, ne;
    int offset_v, offset_e;
    nv = q->nv;
    ne = q->ne;
    n = nv * nm;
    for (i = 0; i < n; i++) o[i] = 0;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(edg2vert(q, sc, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
    for (i = j = 0; i < nm; i++)
        for (k = 0; k < n; k++) {
            d = q->deg[k];
            if (d <= 0) ERR("wrong vert. degree: %d", d);
            if (j >= n) ERR("j=%d >= n=%d", j, n);
            o[j++] /= d;
        }
}

void mesh_scatter_edg2vert(MeshScatter *q, int nm, Scalars *sc, /**/ double *o) {
    int i, n, nv, ne;
    int offset_v, offset_e;
    nv = q->nv;
    ne = q->ne;
    n = nv * nm;
    for (i = 0; i < n; i++) o[i] = 0;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(edg2vert(q, sc, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
}

void mesh_scatter_fin(MeshScatter *q) {
    EFREE(q->ee);
    EFREE(q->deg);
    EFREE(q);
}






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

#ifndef UDX_MESH_DEF_COMMON
#define UDX_MESH_DEF_COMMON
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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
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
#ifndef UDX_EXCH_DEVICE_API_COMMON
#define UDX_EXCH_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif

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
/* 1D texture object template */
static void sz_check(int n) {
    if (n <            0 ) ERR("too small texo: %d", n);
    if (n > MAX_TEXO_SIZE) ERR("too big texo: %d > %d", n, MAX_TEXO_SIZE);
}

template<typename T>
struct Texo {
    cudaTextureObject_t d;
};

template<typename T>
void texo_setup(int n, T *data, /**/ Texo<T> *to)
{
    sz_check(n);

    cudaResourceDesc resD;
    cudaTextureDesc  texD;

    memset(&resD, 0, sizeof(resD));
    resD.resType = cudaResourceTypeLinear;
    resD.res.linear.devPtr = data;
    resD.res.linear.sizeInBytes = n * sizeof(T);
    resD.res.linear.desc = cudaCreateChannelDesc<T>();

    memset(&texD, 0, sizeof(texD));
    texD.normalizedCoords = 0;
    texD.readMode = cudaReadModeElementType;

    CC(cudaCreateTextureObject(&to->d, &resD, &texD, NULL));
}

template<typename T>
void texo_destroy(/**/ Texo<T> *to)
{
    CC(cudaDestroyTextureObject(to->d));
}
template<typename T>
__device__ __forceinline__
const T texo_fetch(const Texo<T> t, const int i)
{
    return Tfetch(T, t.d, i);
}
#ifndef UDX_MESH_TRIANGLES_TYPE
#define UDX_MESH_TRIANGLES_TYPE
struct Triangles {
    int4 *tt;
    int nt;
};
#endif
struct Triangles;
struct MeshRead;

void triangles_ini(MeshRead*, /**/ Triangles**);
void triangles_fin(Triangles*);

#ifndef UDX_EXCH_KL_COMMON
#define UDX_EXCH_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif
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

enum {X, Y, Z};

static __host__ __device__ bool same_side(const float *x, const float *p, const float *a, const float *b, const float *inplane) {
    const float n[3] = {a[Y] * b[Z] - a[Z] * b[Y],
                        a[Z] * b[X] - a[X] * b[Z],
                        a[X] * b[Y] - a[Y] * b[X]};

    const float ndx = n[X] * (x[X] - inplane[X]) + n[Y] * (x[Y] - inplane[Y]) + n[Z] * (x[Z] - inplane[Z]);
    const float ndp = n[X] * (p[X] - inplane[X]) + n[Y] * (p[Y] - inplane[Y]) + n[Z] * (p[Z] - inplane[Z]);

    return ndx * ndp > 0;
}

static __host__ __device__ bool in_tetrahedron(const float *x, const float *A, const float *B, const float *C, const float *D) {
    const float AB[3] = {B[X] - A[X], B[Y] - A[Y], B[Z] - A[Z]};
    const float AC[3] = {C[X] - A[X], C[Y] - A[Y], C[Z] - A[Z]};
    const float AD[3] = {D[X] - A[X], D[Y] - A[Y], D[Z] - A[Z]};

    const float BC[3] = {C[X] - B[X], C[Y] - B[Y], C[Z] - B[Z]};
    const float BD[3] = {D[X] - B[X], D[Y] - B[Y], D[Z] - B[Z]};

    return
        same_side(x, A, BC, BD, D) &&
        same_side(x, B, AD, AC, D) &&
        same_side(x, C, AB, BD, D) &&
        same_side(x, D, AB, AC, A);
}

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt) {
    int c = 0;
    float origin[3] = {0, 0, 0};

    if (spdir != NOT_PERIODIC)
        origin[spdir] = r[spdir];

    for (int i = 0; i < nt; ++i) {
        int4 t = tt[i];
        if (in_tetrahedron(r, vv + 3*t.x, vv + 3*t.y, vv + 3*t.z, origin)) ++c;
    }
    return c%2;
}

namespace collision_dev
{
__global__ void init_tags(const int n, const int color, /**/ int *tags) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < n) tags[gid] = color;
}

union Pos {
    float2 f2[2];
    struct { float r[3]; float dummy; };
};

static __device__ Pos fetchPos(const Particle *vert, const int id) {
    const float2 *pp = (const float2*) vert;
    Pos r;
    r.f2[0] = pp[3 * id + 0];
    r.f2[1] = pp[3 * id + 1];
    return r;
}

static __device__ bool inside_box(const float r[3], float3 lo, float3 hi) {
    enum {X, Y, Z};
    return
        r[X] >= lo.x && r[X] <= hi.x &&
        r[Y] >= lo.y && r[Y] <= hi.y &&
        r[Z] >= lo.z && r[Z] <= hi.z;
}

/* assume nm blocks along y */
__global__ void label(int pdir, const Particle *pp, const int n, const Particle *vert, const int nv,
                      Triangles tri, const float3 *minext, const float3 *maxext,
                      int lab_in, /**/ int *labels) {
    int i, sid, gid, count, mbase;
    Particle p;
    Pos a, b, c;
    float3 lo, hi;
    int4 t;
    sid = blockIdx.y;
    gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= n) return;

    count = 0;

    p = pp[gid];

    lo = minext[sid];
    hi = maxext[sid];
    if (!inside_box(p.r, lo, hi)) return;

    float origin[3] = {0, 0, 0};
    if (pdir != NOT_PERIODIC) origin[pdir] = p.r[pdir];

    mbase = nv * sid;
    for (i = 0; i < tri.nt; ++i) {
        t = tri.tt[i];

        a = fetchPos(vert, mbase + t.x);
        b = fetchPos(vert, mbase + t.y);
        c = fetchPos(vert, mbase + t.z);

        if (in_tetrahedron(p.r, a.r, b.r, c.r, origin)) ++count;
    }

    // dont consider the case of inside several solids
    if (count % 2) atomicExch(labels + gid, lab_in);
}
}

void collision_label_ini(long n, int lab_out, /**/ int *labels) {
    KL(collision_dev::init_tags, (k_cnf(n)), (n, lab_out, /**/ labels));
}

void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri,
                     int nv, int nm, const Particle *i_pp,
                     const float3 *minext, const float3 *maxext,
                     int lab_in, /**/ int *labels) {
    enum {THR = 128};

    dim3 thrd(THR, 1);
    dim3 blck(ceiln(n, THR), nm);

    if (nm == 0 || n == 0) return;

    KL(collision_dev::label, (blck, thrd),
       (pdir, pp, n, i_pp, nv, *tri, minext, maxext, lab_in, /**/ labels));

}






// Grouped editable compilation unit for mesh coordinates, distance, generation, and energy.


// Grouped editable compilation unit for mesh coordinates and distance properties.


// Grouped editable compilation unit for cylindrical and spherical coordinates.



#include <stdio.h>
#include <math.h>

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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

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
struct KahanSum;

void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>

struct MeshCylindrical;
struct Vectors;

void mesh_cylindrical_ini(int nv, /**/ MeshCylindrical**);
void mesh_cylindrical_fin(MeshCylindrical*);
void  mesh_cylindrical_apply(MeshCylindrical*, int nm, Vectors*, /**/
                           double *r, double *phi, double *z);
#include "udx.mesh-coordinate-common.cuh"
struct MeshCylindrical {
    double *rr; /* workspace for coordinates shifted to a mesh center
                   of mass */
    int nv;
};
void mesh_cylindrical_ini(int nv, MeshCylindrical **pq) {
    MeshCylindrical *q;
    EMALLOC(1, &q);
    EMALLOC(3*nv, &q->rr);
    q->nv = nv;
    *pq = q;
}

void mesh_cylindrical_fin(MeshCylindrical *q) {
    EFREE(q->rr);
    EFREE(q);
}

static void xyz2cyl(double x, double y, double z, /**/ double *pr, double *pphi, double *pz) {
    double r, phi;
    r = sqrt(x*x + y*y);
    phi = atan2(y, x);
    *pr = r; *pphi  = phi; *pz = z;
}
static void mesh_cylindrical_apply_coordinates(int n, double *rr, /**/ double *r, double *phi, double *z) {
    enum {X, Y, Z};
    int i;
    for (i = 0; i < n; i++) {
        xyz2cyl(rr[X], rr[Y], rr[Z], /**/ r, phi, z);
        rr += 3; r++; phi++; z++;
    }
}
void mesh_cylindrical_apply(MeshCylindrical *q, int m, Vectors *pos, /**/ double *r, double *phi, double *z) {
    int i, nv, offset;
    double *rr;
    nv = q->nv; rr = q->rr; offset = 0;
    for (i = 0; i < m; i++) {
        UC(mesh_coordinate_to_com(nv, offset, pos, /**/ rr));
        mesh_cylindrical_apply_coordinates(nv, rr, /**/ r, phi, z);
        rr += 3*nv; r += nv; phi += nv; z += nv; offset += nv;
    }
}





#include <stdio.h>
#include <math.h>

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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>

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
struct KahanSum;

void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>

struct MeshSpherical;
struct Vectors;

void mesh_spherical_ini(int nv, /**/ MeshSpherical**);
void mesh_spherical_fin(MeshSpherical*);
void  mesh_spherical_apply(MeshSpherical*, int nm, Vectors*, /**/
                           double *r, double *theta, double *phi);
#include "udx.mesh-coordinate-common.cuh"
struct MeshSpherical {
    double *rr; /* workspace for coordinates shifted to a mesh center
                   of mass */
    int nv;
};
void mesh_spherical_ini(int nv, MeshSpherical **pq) {
    MeshSpherical *q;
    EMALLOC(1, &q);
    EMALLOC(3*nv, &q->rr);
    q->nv = nv;
    *pq = q;
}

void mesh_spherical_fin(MeshSpherical *q) {
    EFREE(q->rr);
    EFREE(q);
}

static void xyz2sph(double x, double y, double z, /**/ double *pr, double *ptheta, double *pphi) {
    double r, theta, phi;
    r = sqrt(x*x + y*y + z*z);
    theta = atan2(z, r);
    phi = atan2(y, x);
    *pr = r; *ptheta = theta; *pphi = phi;
}
static void mesh_spherical_apply_coordinates(int n, double *rr, /**/ double *r, double *theta, double *phi) {
    enum {X, Y, Z};
    int i;
    for (i = 0; i < n; i++) {
        xyz2sph(rr[X], rr[Y], rr[Z], /**/ r, theta, phi);
        rr += 3; r++; theta++; phi++;
    }
}
void mesh_spherical_apply(MeshSpherical *q, int m, Vectors *pos, /**/ double *r, double *theta, double *phi) {
    int i, nv, offset;
    double *rr;
    nv = q->nv; rr = q->rr; offset = 0;
    for (i = 0; i < m; i++) {
        UC(mesh_coordinate_to_com(nv, offset, pos, /**/ rr));
        mesh_spherical_apply_coordinates(nv, rr, /**/ r, theta, phi);
        rr += 3*nv; r += nv; theta += nv; phi += nv; offset += nv;
    }
}





// Grouped editable compilation unit for mesh distance and properties.



#include <vector_types.h>
#include <math.h>

#include "udx.mesh-common.cuh"
struct int4;
float dist_from_mesh(int nt, const int4 *tt, const float *vv, const float *r0);

static float fmin_local(float a, float b) {return a < b ? a : b;}
static float fmax_local(float a, float b) {return a < b ? b : a;}
static float dot(const float x[3], const float y[3]) { return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]; }

static void project_t(const float *a, const float *b, const float *c, const float *r, /**/ float *p) {
    const float ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    const float ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
    const float ar[3] = {r[0]-a[0], r[1]-a[1], r[2]-a[2]};

    float n[3] = {ab[1]*ac[2] - ab[2]*ac[1],
                  ab[2]*ac[0] - ab[0]*ac[2],
                  ab[0]*ac[1] - ab[1]*ac[0]};
    {
        const float s = 1.f / sqrt(dot(n,n));
        n[0] *= s; n[1] *= s; n[2] *= s;
    }

    const float arn = dot(ar, n);
    const float g[3] = {ar[0] - arn * n[0],
                        ar[1] - arn * n[1],
                        ar[2] - arn * n[2]};

    float u, v;
    {
        const float ga1 = dot(g, ab);
        const float ga2 = dot(g, ac);
        const float a11 = dot(ab, ab);
        const float a12 = dot(ab, ac);
        const float a22 = dot(ac, ac);

        const float fac = 1.f / (a11*a22 - a12*a12);

        u = (ga1 * a22 - ga2 * a12) * fac;
        v = (ga2 * a11 - ga1 * a12) * fac;
    }

    // project (u,v) onto unit triangle

    if ( (v > u - 1) && (v < u + 1) && (v > 1 - u) ) {
        const float a_ = 0.5f * (u + v - 1);
        u -= a_;
        v -= a_;
    }
    else {
        u = fmax_local(fmin_local(1.f, u), 0.f);
        v = fmax_local(fmin_local(v, 1.f-u), 0.f);
    }

    // compute projected point
    p[0] = a[0] + u * ab[0] + v * ac[0];
    p[1] = a[1] + u * ab[1] + v * ac[1];
    p[2] = a[2] + u * ab[2] + v * ac[2];
}

static float dist_from_triangle(const float *a, const float *b, const float *c, const float *r) {
    float p[3];
    project_t(a, b, c, r, /**/ p);

    const float dr[3] = {p[0] - r[0], p[1] - r[1], p[2] - r[2]};
    return sqrt(dot(dr, dr));
}

float dist_from_mesh(int nt, const int4 *tt, const float *vv, const float *r0) {
    float dmin = 1e5f;

    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];
        const int i1 = t.x;
        const int i2 = t.y;
        const int i3 = t.z;

        const float *A = vv + 3*i1;
        const float *B = vv + 3*i2;
        const float *C = vv + 3*i3;

        const float d = dist_from_triangle(A, B, C, r0);

        dmin = fmin_local(d, dmin);
    }
    return dmin;
}





#include <vector_types.h>
#include <string.h>

#include "udx.mesh-common.cuh"
struct int4;
float mesh_volume0(int nt, const int4 *tt, const float *vv);
void mesh_center_of_mass(int nt, const int4 *tt, const float *vv, /**/ float *com);
void mesh_inertia_tensor(int nt, const int4 *tt, const float *vv, const float *com, const float density, /**/ float *I);

enum {XX, XY, XZ, YY, YZ, ZZ};

#define Ax A[X]
#define Ay A[Y]
#define Az A[Z]

#define Bx B[X]
#define By B[Y]
#define Bz B[Z]

#define Cx C[X]
#define Cy C[Y]
#define Cz C[Z]

/* see http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf */

static void M_0(const float *A, const float *B, const float *C, /**/ float *res) {
    *res =  (+Ax * (By * Cz -Bz * Cy)
             -Ay * (Bx * Cz -Bz * Cx)
             +Az * (Bx * Cy -By * Cx)) / 6.f;
}

static void M_1(const float *A, const float *B, const float *C, /**/ float *res) {
    const float common = (+Ax * (By * Cz -Bz * Cy)
                          -Ay * (Bx * Cz -Bz * Cx)
                          +Az * (Bx * Cy -By * Cx)) / 24.f;

    res[0] = common * (Ax + Bx + Cx);
    res[1] = common * (Ay + By + Cy);
    res[2] = common * (Az + Bz + Cz);
}

static void M_2(const float *A, const float *B, const float *C, /**/ float *res) {
    /* see /poc/mesh/moments.mac */
    const float D1 = (+Ax * (By*Cz - Bz*Cy)
                      -Ay * (Bx*Cz - Bz*Cx)
                      +Az * (Bx*Cy - By*Cx)) / 60.f;
    const float D2 = By+Ay;
    const float D3 = Bz+Az;
    const float D4 = 2*Bx+Ax;
    const float D5 = 2*Cx+Bx+Ax;
    const float D6 = 0.5f * D1;

    res[XX] = (Ax*Ax + Ax*Bx + Bx*Bx + (Bx+Ax)*Cx + Cx*Cx) * D1;
    res[YY] = (Ay*Ay + Ay*By + By*By +    D2 * Cy + Cy*Cy) * D1;
    res[ZZ] = (Az*Az + Az*Bz + Bz*Bz +    D3 * Cz + Cz*Cz) * D1;

    res[XY] = (2 * Ax*Ay + Ay*Bx +        D4*By + D2*Cx +           D5*Cy) * D6;
    res[XZ] = (2 * Ax*Az + Az*Bx +        D4*Bz + D3*Cx +           D5*Cz) * D6;
    res[YZ] = (2 * Ay*Az + Az*By + (2*By+Ay)*Bz + D3*Cy + (2*Cy+By+Ay)*Cz) * D6;
}

#undef Ax
#undef Ay
#undef Az

#undef Bx
#undef By
#undef Bz

#undef Cx
#undef Cy
#undef Cz

#define load_t(vv, tid) {vv[3*tid + 0], vv[3*tid + 1], vv[3*tid + 2]}

float mesh_volume0(int nt, const int4 *tt, const float *vv) {
    float Vtot = 0;

    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];

        const float A[3] = load_t(vv, t.x);
        const float B[3] = load_t(vv, t.y);
        const float C[3] = load_t(vv, t.z);

        float V = 0;
        M_0(A, B, C, /**/ &V);

        Vtot += V;
    }
    return Vtot;
}

void mesh_center_of_mass(int nt, const int4 *tt, const float *vv, /**/ float *com) {
    float Vtot = 0, M1tot[3] = {0};

    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];

        const float A[3] = load_t(vv, t.x);
        const float B[3] = load_t(vv, t.y);
        const float C[3] = load_t(vv, t.z);

        float V = 0, M1[3];

        M_0(A, B, C, /**/ &V);
        M_1(A, B, C, /**/ M1);

        Vtot += V;
        M1tot[X] += M1[X];
        M1tot[Y] += M1[Y];
        M1tot[Z] += M1[Z];
    }

    com[X] = M1tot[X] / Vtot;
    com[Y] = M1tot[Y] / Vtot;
    com[Z] = M1tot[Z] / Vtot;
}

static void shift(const float s[3], /**/ float a[3]) {
    a[X] -= s[X]; a[Y] -= s[Y]; a[Z] -= s[Z];
}

void mesh_inertia_tensor(int nt, const int4 *tt, const float *vv, const float *com, const float density, /**/ float *I) {
    memset(I, 0, 6 * sizeof(float));

    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];

        float A[3] = load_t(vv, t.x);
        float B[3] = load_t(vv, t.y);
        float C[3] = load_t(vv, t.z);

        shift(com, /**/ A);
        shift(com, /**/ B);
        shift(com, /**/ C);

        float M2[6];
        M_2(A, B, C, /**/ M2);

        I[XX] += M2[YY] + M2[ZZ];
        I[YY] += M2[XX] + M2[ZZ];
        I[ZZ] += M2[XX] + M2[YY];

        I[XY] -= M2[XY];
        I[XZ] -= M2[XZ];
        I[YZ] -= M2[YZ];
    }

    for (int c = 0; c < 6; ++c)
        I[c] *= density;
}

#undef load_t






// Grouped editable compilation unit for mesh generation, energies, and forces.


// Grouped editable compilation unit for mesh energies and Kantor force backends.


// Grouped editable compilation unit for Julicher and Kantor mesh energies.



#include <stdio.h>

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

namespace tri_hst {
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
}

struct MeshEdgLen;
struct MeshRead;
struct Vectors;

void mesh_edg_len_ini(MeshRead*, MeshEdgLen**); // <1>
void mesh_edg_len_fin(MeshEdgLen*);
void mesh_edg_len_apply(MeshEdgLen*, int nm, Vectors*, /**/ double*); // <2>
struct MeshAngle;
struct Vectors;
struct MeshRead;

void mesh_angle_ini(MeshRead*, /**/ MeshAngle**); // <1>
void mesh_angle_fin(MeshAngle*);
void  mesh_angle_apply(MeshAngle*, int nm, Vectors*, /**/ double*); // <2>
struct MeshVertArea;
struct Vectors;
struct MeshRead;

void mesh_vert_area_ini(MeshRead*, /**/ MeshVertArea**); // <1>
void mesh_vert_area_fin(MeshVertArea*);

void  mesh_vert_area_apply(MeshVertArea*, int nm, Vectors*, /**/ double*); // <2>
struct MeshScatter;
struct MeshRead;
struct Scalars;

void mesh_scatter_ini(MeshRead*, MeshScatter**); // <1>
void mesh_scatter_fin(MeshScatter*);
void mesh_scatter_edg2vert(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
void mesh_scatter_edg2vert_avg(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>

struct MeshEngJulicher;
struct Vectors;
struct MeshRead;

void mesh_eng_julicher_ini(MeshRead*, int max_nm, /**/ MeshEngJulicher**); // <1>
void mesh_eng_julicher_fin(MeshEngJulicher*);

void  mesh_eng_julicher_apply(MeshEngJulicher*, int nm, Vectors*, double kb, /**/ double*); // <2>
struct MeshEngJulicher {
    int max_nm; /* maximum number of meshes */
    int nv, ne;

    /* edge lens, edge angles, vert. areas, vert. curvatures */
    double *lens, *angles, *areas, *curvs_edg, *curvs_vert;

    MeshEdgLen *len;
    MeshAngle  *angle;
    MeshVertArea *area;

    MeshScatter *scatter;
};
void mesh_eng_julicher_ini(MeshRead *mesh, int nm, /**/ MeshEngJulicher **pq) {
    int nv, ne;
    MeshEngJulicher *q;
    EMALLOC(1, &q);

    UC(nv = mesh_read_get_nv(mesh));
    UC(ne = mesh_read_get_ne(mesh));

    q->max_nm = nm;
    q->nv = nv; q->ne = ne;
    EMALLOC(nm*ne, &q->lens);
    EMALLOC(nm*ne, &q->angles);
    EMALLOC(nm*nv, &q->areas);
    EMALLOC(nm*ne, &q->curvs_edg);
    EMALLOC(nm*nv, &q->curvs_vert);

    mesh_edg_len_ini(mesh, /**/ &q->len);
    mesh_angle_ini(mesh, /**/ &q->angle);
    mesh_vert_area_ini(mesh, /**/ &q->area);
    mesh_scatter_ini(mesh, /**/ &q->scatter);

    *pq = q;
}

void mesh_eng_julicher_fin(MeshEngJulicher *q) {
    EFREE(q->lens);
    EFREE(q->angles);
    EFREE(q->areas);
    EFREE(q->curvs_edg);
    EFREE(q->curvs_vert);

    mesh_edg_len_fin(q->len);
    mesh_angle_fin(q->angle);
    mesh_vert_area_fin(q->area);
    mesh_scatter_fin(q->scatter);

    EFREE(q);
}

void mesh_eng_julicher_apply(MeshEngJulicher *q, int nm, Vectors *pos, double kb, /**/ double *o) {
    int i, nv, ne;
    double *lens, *angles, *areas, *curvs_vert, *curvs_edg;
    double area, curv;
    Scalars *sc;
    lens = q->lens; angles = q->angles; areas = q->areas;
    curvs_edg = q->curvs_edg; curvs_vert = q->curvs_vert;

    ne = q->ne; nv = q->nv;
    mesh_edg_len_apply(q->len, nm, pos, /**/ lens);
    mesh_angle_apply(q->angle, nm, pos, /**/ angles);
    for (i = 0; i < nm * ne; i++)
        curvs_edg[i] = lens[i]*angles[i];

    scalars_double_ini(nm * ne, curvs_edg, /**/ &sc);
    mesh_scatter_edg2vert(q->scatter, nm, sc, /**/ curvs_vert);
    for (i = 0; i < nm * nv; i++)
        curvs_vert[i] /= 4;

    mesh_vert_area_apply(q->area, nm, pos, /**/ areas);
    for (i = 0; i < nm * nv; i++) {
        curv = curvs_vert[i];
        area = areas[i];
        if (area <= 0) ERR("area[%d]=%g <= 0", area, i);
        o[i] = 2*kb*curv*curv/area;
    }
    scalars_fin(sc);
    mesh_vert_area_apply(q->area, nm, pos, /**/ areas);
}





#include <stdio.h>
#include <math.h>

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

namespace tri_hst {
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
}
struct MeshAngle;
struct Vectors;
struct MeshRead;

void mesh_angle_ini(MeshRead*, /**/ MeshAngle**); // <1>
void mesh_angle_fin(MeshAngle*);
void  mesh_angle_apply(MeshAngle*, int nm, Vectors*, /**/ double*); // <2>

struct MeshEngKantor;
struct Vectors;
struct MeshRead;

void mesh_eng_kantor_ini(MeshRead*, int max_nm, /**/ MeshEngKantor**); // <1>
void mesh_eng_kantor_fin(MeshEngKantor*);

void  mesh_eng_kantor_apply(MeshEngKantor*, int nm, Vectors*, double kb, double ang, /**/ double*); // <2>
struct MeshEngKantor {
    int max_nm; /* maximum number of meshes */
    int nv, ne;
    double *angles;
    MeshAngle  *angle;
};
void mesh_eng_kantor_ini(MeshRead *mesh, int nm, /**/ MeshEngKantor **pq) {
    int nv, ne;
    MeshEngKantor *q;
    EMALLOC(1, &q);

    UC(nv = mesh_read_get_nv(mesh));
    UC(ne = mesh_read_get_ne(mesh));

    q->max_nm = nm;
    q->nv = nv; q->ne = ne;
    EMALLOC(nm*ne, &q->angles);
    mesh_angle_ini(mesh, /**/ &q->angle);

    *pq = q;
}

void mesh_eng_kantor_fin(MeshEngKantor *q) {
    EFREE(q->angles);
    mesh_angle_fin(q->angle);
    EFREE(q);
}

void mesh_eng_kantor_apply(MeshEngKantor *q, int nm, Vectors *pos, double kb, double angle0, /**/ double *o) {
    int i, ne;
    double *angles, angle;
    angles = q->angles;
    ne = q->ne;
    mesh_angle_apply(q->angle, nm, pos, /**/ angles);
    for (i = 0; i < nm * ne; i++) {
        angle = angles[i];
        o[i] = 2*kb*(1 - cos(angle - angle0));
    }
}





// Grouped editable compilation unit for Kantor force backends.


#include "udx.mesh-force-kantor-common.cuh"

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <assert.h>
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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
struct float3;
struct double3;
namespace force_kantor0_hst {
double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
}

#define FORCE_KANTOR0_HOST
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
#undef FORCE_KANTOR0_HOST




#include "udx.mesh-force-kantor-common.cuh"

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <assert.h>
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
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
struct double3;
namespace force_kantor1_hst {
double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
}

#define FORCE_KANTOR1_HOST
#ifdef FORCE_KANTOR1_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_kantor1_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor1_dev {
  #define END }
#endif

BEGIN

#ifdef FORCE_KANTOR1_HOST
#define PRINT(fmt, ...) msg_print((fmt), ##__VA_ARGS__)
#define EXIT() ERR("assert")
#else
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)
#endif

_S_ void dih_a0(int Flag_a,
                double ax, double ay, double az,
                double bx, double by, double bz,
                double cx, double cy, double cz,
                double dx, double dy, double dz,
                double *pfx, double *pfy, double *pfz) {
    double E_ax,E_ay,E_az,E_bx,E_by,E_bz,E_kk,E_kn,E_kx,
        E_ky,E_kz,E_nn,E_nx,E_ny,E_nz,E_rsq,kk,
        kk_kx,kk_ky,kk_kz,kn_kx,kn_ky,kn_kz,kn_nx,kn_ny,kn_nz,kx,kx_ay,kx_az,kx_by,
        kx_bz,ky,ky_ax,ky_az,ky_bx,ky_bz,kz,kz_ax,kz_ay,
        kz_bx,kz_by,nn,nn_nx,nn_ny,nn_nz,nx,nx_by,nx_bz,
        ny,ny_bx,ny_bz,nz,nz_bx,nz_by,rsq_kk,rsq_nn,rsq,kn;

    E_ax=E_ay=E_az=E_bx=E_by=E_bz=E_kk=E_kn=E_kx=
        E_ky=E_kz=E_nn=E_nx=E_ny=E_nz=E_rsq=kk=0.0;
    kz = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
    ky = (bz-az)*(cx-ax)-(bx-ax)*(cz-az);
    kx = (by-ay)*(cz-az)-(bz-az)*(cy-ay);
    nz = (cy-by)*(dx-bx)-(cx-bx)*(dy-by);
    ny = (cx-bx)*(dz-bz)-(cz-bz)*(dx-bx);
    nx = (cz-bz)*(dy-by)-(cy-by)*(dz-bz);

    nn = pow(nz,2)+pow(ny,2)+pow(nx,2);
    kk = pow(kz,2)+pow(ky,2)+pow(kx,2);
    rsq = 1/sqrt(kk*nn);
    kn = kz*nz+ky*ny+kx*nx;
    E_kn = rsq;
    E_rsq = kn;
    kn_kx = nx;
    kn_ky = ny;
    kn_kz = nz;
    kn_nx = kx;
    kn_ny = ky;
    kn_nz = kz;
    E_kx += E_kn*kn_kx;
    E_ky += E_kn*kn_ky;
    E_kz += E_kn*kn_kz;
    E_nx += E_kn*kn_nx;
    E_ny += E_kn*kn_ny;
    E_nz += E_kn*kn_nz;
    rsq_kk = -(nn*pow(kk*nn,(-3.0E+0)/2.0E+0))/2.0E+0;
    rsq_nn = -(kk*pow(kk*nn,(-3.0E+0)/2.0E+0))/2.0E+0;
    E_kk += E_rsq*rsq_kk;
    E_nn += E_rsq*rsq_nn;
    kk_kx = 2*kx;
    kk_ky = 2*ky;
    kk_kz = 2*kz;
    E_kx += E_kk*kk_kx;
    E_ky += E_kk*kk_ky;
    E_kz += E_kk*kk_kz;
    nn_nx = 2*nx;
    nn_ny = 2*ny;
    nn_nz = 2*nz;
    E_nx += E_nn*nn_nx;
    E_ny += E_nn*nn_ny;
    E_nz += E_nn*nn_nz;
    nx_by = dz-cz;
    nx_bz = cy-dy;
    E_by += E_nx*nx_by;
    E_bz += E_nx*nx_bz;
    ny_bx = cz-dz;
    ny_bz = dx-cx;
    E_bx += E_ny*ny_bx;
    E_bz += E_ny*ny_bz;
    nz_bx = dy-cy;
    nz_by = cx-dx;
    E_bx += E_nz*nz_bx;
    E_by += E_nz*nz_by;
    kx_ay = bz-cz;
    kx_az = cy-by;
    kx_by = cz-az;
    kx_bz = ay-cy;
    E_ay += E_kx*kx_ay;
    E_az += E_kx*kx_az;
    E_by += E_kx*kx_by;
    E_bz += E_kx*kx_bz;
    ky_ax = cz-bz;
    ky_az = bx-cx;
    ky_bx = az-cz;
    ky_bz = cx-ax;
    E_ax += E_ky*ky_ax;
    E_az += E_ky*ky_az;
    E_bx += E_ky*ky_bx;
    E_bz += E_ky*ky_bz;
    kz_ax = by-cy;
    kz_ay = cx-bx;
    kz_bx = cy-ay;
    kz_by = ax-cx;
    E_ax += E_kz*kz_ax;
    E_ay += E_kz*kz_ay;
    E_bx += E_kz*kz_bx;
    E_by += E_kz*kz_by;

    if (Flag_a) {
        *pfx = E_ax; *pfy = E_ay; *pfz = E_az;
    } else {
        *pfx = E_bx; *pfy = E_by; *pfz = E_bz;
    }
}

_I_ double3 dih_a(double, double kb,
                  double3 a, double3 b, double3 c, double3 d) {
    double3 f;
    double fx, fy, fz;
    int Flag_a;
    Flag_a = 1;
    dih_a0(Flag_a,
           a.x, a.y, a.z, b.x, b.y, b.z,
           c.x, c.y, c.z, d.x, d.y, d.z,
           &fx, &fy, &fz);
    f.x = kb*fx; f.y = kb*fy; f.z = kb*fz;
    return f;
}

_I_ double3 dih_b(double, double kb,
                  double3 a, double3 b, double3 c, double3 d) {
    double3 f;
    double fx, fy, fz;
    int Flag_a;
    Flag_a = 0;
    dih_a0(Flag_a,
           a.x, a.y, a.z, b.x, b.y, b.z,
           c.x, c.y, c.z, d.x, d.y, d.z,
           &fx, &fy, &fz);
    f.x = kb*fx; f.y = kb*fy; f.z = kb*fz;
    return f;
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
#undef FORCE_KANTOR1_HOST






// Grouped editable compilation unit for matrix and mesh generation.



#include <stdio.h>
#include <string.h>

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

struct Matrices;
struct Coords;

void matrices_read(const char *path, /**/ Matrices**); // <1>
void matrices_read_filter(const char *path, const Coords*, /**/ Matrices**); // <2>

void matrices_get(const Matrices*, int i, /**/ double**); // <3>
void matrices_get_r(const Matrices*, int i, /**/ double r[3]); // <4>
int  matrices_get_n(const Matrices*); // <5>
void matrices_log(const Matrices*);   // <6>

void matrices_fin(Matrices*);

struct Matrix { double D[4*4]; };
struct Matrices {
    int n;
    Matrix *m;
};
static const int MAX_N = 100000;

static void matrix2r(const double A[16], double r[3]) {
    enum {X, Y, Z};
    int i;
    i = 0;
    i++; i++; i++; r[X] = A[i++];
    i++; i++; i++; r[Y] = A[i++];
    i++; i++; i++; r[Z] = A[i++];
}

static int read_matrix(FILE *f, double A[16]) {
    int i;
    for (i = 0; i < 16; i++)
        if (fscanf(f, "%lf", &A[i]) != 1) return 0;
    return 1;
}

void matrices_read(const char *path, /**/ Matrices **pq) {
    int n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    msg_print("read '%s'", path);
    UC(efopen(path, "r", /**/ &f));
    n = 0;
    while (read_matrix(f, /**/ q->m[n].D)) {
        n++;
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    }
    UC(efclose(f));
    q->n = n;
    *pq = q;
}

static int good(const Coords *c, const double A[16]) {
    enum {X, Y, Z};
    double r[3];
    matrix2r(A, /**/ r);
    return
        xlo(c) <= r[X] && r[X] < xhi(c) &&
        ylo(c) <= r[Y] && r[Y] < yhi(c) &&
        zlo(c) <= r[Z] && r[Z] < zhi(c);
}
void matrices_read_filter(const char *path, const Coords *c, /**/ Matrices **pq) {
    int m, n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    msg_print("read '%s'", path);
    UC(efopen(path, "r", /**/ &f));
    n = m = 0;
    while (read_matrix(f, /**/ q->m[n].D)) {
        m++;
        if (good(c, q->m[n].D)) n++;
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    }
    msg_print("took %d/%d matrices", n, m);
    UC(efclose(f));
    q->n = n;
    *pq = q;
}

void matrices_get(const Matrices *q, int i, /**/ double **pq) {
    int n;
    n = q->n;
    if (i >= n) ERR("i=%d >= n=%d", i, n);
    *pq = q->m[i].D;
}

void matrices_get_r(const Matrices *q, int k, /**/ double r[3]) {
    enum {X, Y, Z};
    int n;
    double *A;
    n = q->n;
    if (k >= n) ERR("k=%d >= n=%d", k, n);
    A = q->m[k].D;
    matrix2r(A, /**/ r);
}

int matrices_get_n(const Matrices *q) { return q->n; }

static void log(double A[16]) {
    int i, j;
    double a, b, c, d;
    for (i = j = 0; j < 4; j++) {
        a = A[i++]; b = A[i++]; c = A[i++]; d = A[i++];
        msg_print("%g %g %g %g", a, b, c, d);
    }
}
void matrices_log(const Matrices *q) {
    int n, i;
    n = q->n;
    msg_print("<matrices_log");
    for (i = 0; i < n; i++) {
        if (i > 0) msg_print("");
        log(q->m[i].D);
    }
    msg_print(">matrices_log");
}

void matrices_fin(Matrices *q) { EFREE(q->m); EFREE(q); }





#include <stdio.h>
#include <assert.h>
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

#ifndef UDX_MESH_DEF_COMMON
#define UDX_MESH_DEF_COMMON
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
#endif
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
#endif /* UDX_EXCH_TYPE_COMMON */

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


struct Matrices;
struct Coords;

void matrices_read(const char *path, /**/ Matrices**); // <1>
void matrices_read_filter(const char *path, const Coords*, /**/ Matrices**); // <2>

void matrices_get(const Matrices*, int i, /**/ double**); // <3>
void matrices_get_r(const Matrices*, int i, /**/ double r[3]); // <4>
int  matrices_get_n(const Matrices*); // <5>
void matrices_log(const Matrices*);   // <6>

void matrices_fin(Matrices*);
struct Coords;
struct Particle;
struct Matrices;

/* vv : vertices of the mesh template : x y z x y z, ...   */
int  mesh_gen_from_file(const Coords*, const float *vv, const char *ic, int nv, /**/ Particle*); // <1>
void mesh_gen_from_matrices(int nv, const float *vv, const Matrices*, /**/ int *pn, Particle*);  // <2>
void mesh_shift(const Coords*, int n, /**/ Particle*); // <3>

static void mult(double A[16], const double a[4], /**/ double b[4]) {
    /* b = A x a */
    enum {X, Y, Z, W};
    int c, i;
    for (i = c = 0; c < 4; c++) {
        b[c]  = A[i++]*a[X];
        b[c] += A[i++]*a[Y];
        b[c] += A[i++]*a[Z];
        b[c] += A[i++]*a[W];
    }
}

static void homogenius_multi(double A[16], const float a0[3], /**/ float b0[3]) {
    /* b = A x [a, 1] */
    enum {X, Y, Z, W};
    double a[4], b[4];
    a[X] = a0[X]; a[Y] = a0[Y]; a[Z] = a0[Z]; a[W] = 1;
    mult(A, a, /**/ b);
    b0[X] = b[X]; b0[Y] = b[Y]; b0[Z] = b[Z];
}

static void gen0(double A[16], const float *r, Particle *p) {
    enum {X, Y, Z};
    p->v[X] = p->v[Y] = p->v[Z] = 0;
    homogenius_multi(A, r, /**/ p->r);
}

static void gen1(double A[16], int nv, const float *rr, Particle *pp) {
    for (int i = 0; i < nv; i++) gen0(A, &rr[3*i], &pp[i]);
}

void mesh_gen_from_matrices(int nv, const float *rr, const Matrices *matrices, /**/ int *pn, Particle *pp) {
    int i, n, nm;
    double *A;
    n = 0;
    nm = matrices_get_n(matrices);
    for (i = 0; i < nm; i++) {
        matrices_get(matrices, i, &A);
        gen1(A, nv, rr, &pp[n]); n += nv;
    }
    *pn = n;
}

static void shift(const Coords *c, Particle *p) {
    enum {X, Y, Z};
    float *r;
    r = p->r;
    r[X] = xg2xl(c, r[X]);
    r[Y] = yg2yl(c, r[Y]);
    r[Z] = zg2zl(c, r[Z]);
}

void mesh_shift(const Coords *c, int n, Particle *pp) {
    for (int i = 0; i < n; i++) shift(c, &pp[i]);
}

int mesh_gen_from_file(const Coords *coords, const float *rr0, const char *path, int nv, Particle *pp) {
    int n;
    Matrices *matrices;
    if (nv <= 0) ERR("nv <= 0");
    UC(matrices_read_filter(path, coords, /**/ &matrices));
    UC(mesh_gen_from_matrices(nv, rr0, matrices, /**/ &n, pp));
    UC(mesh_shift(coords, n, pp));
    UC(matrices_fin(matrices));
    return n / nv;
}









#include <assert.h>
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
#ifndef UDX_EXCH_KL_COMMON
#define UDX_EXCH_KL_COMMON
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_NONE)
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#elif  defined(KL_SYNC)
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#else
  #error KL_* is undefined
#endif /* UDX_EXCH_KL_COMMON */

namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* TODO: header includes header */
#endif
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
#endif /* UDX_EXCH_TYPE_COMMON */
#ifndef UDX_EXCH_DEVICE_API_COMMON
#define UDX_EXCH_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}
#endif /* UDX_EXCH_DEVICE_API_COMMON */

#ifndef UDX_MESH_FORCE_KANTOR_COMMON_CUH
#define UDX_MESH_FORCE_KANTOR_COMMON_CUH
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
#endif

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
struct MeshBB {
    int *ncols;       /* number of possible collisions per particle      */
    float4 *datacol;  /* list of data related to collisions per particle */
    int *idcol;       /* list of triangle colliding ids per particle     */
};

/* conf */
enum {MAX_COL = 4};

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

#define _I_ static __device__
#define _S_ static __device__

namespace mesh_bounce_dev {
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
typedef double real;
typedef double3 real3;
#ifndef UDX_MESH_COORD_COMMON
#define UDX_MESH_COORD_COMMON
enum {X, Y, Z};
#endif


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
static __device__ bool cubic_root(real dt, real a, real b, real c, real d, /**/ real *h) {
    const real eps = 1e-8;
    real h1, h2, h3;

    if (fabs(a) > eps) { // cubic
        b /= a;
        c /= a;
        d /= a;

        int nsol = cubic(b, c, d, &h1, &h2, &h3);

        if (valid_time(dt, h1))             {*h = h1; return true;}
        if (nsol > 1 && valid_time(dt, h2)) {*h = h2; return true;}
        if (nsol > 2 && valid_time(dt, h3)) {*h = h3; return true;}
    }
    else if (fabs(b) > eps) { // quadratic
        if (!quadratic(b, c, d, &h1, &h2)) return false;
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
}

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

#undef _I_
#undef _S_





// Grouped editable compilation unit for particle, force, and paired arrays.


// Grouped editable compilation unit for the particle and force array views.



#include <stdio.h>

#ifndef UDX_STRUCT_PARRAY_COMMON
#define UDX_STRUCT_PARRAY_COMMON
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

struct PaArray_v {
    const float *pp;
};

struct PaCArray_v {
    const float *pp;
    const int *cc;   /* colors */
};
void parray_push_pp(const Particle *pp, PaArray *a) {
    a->pp = (const float*)pp;
    a->colors = false;
};

void parray_push_cc(const int *cc, PaArray *a) {
    a->cc = cc;
    a->colors = true;
}

bool parray_is_colored(const PaArray *a) {
    return a->colors;
}

void parray_get_view(const PaArray *a, PaArray_v *v) {
    v->pp = a->pp;
}

void parray_get_view(const PaArray *a, PaCArray_v *v) {
    v->pp = a->pp;
    v->cc = a->cc;
}






#include <stdio.h>

#ifndef UDX_STRUCT_FARRAY_COMMON
#define UDX_STRUCT_FARRAY_COMMON
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

struct FoArray_v {
    float *ff;
};

struct FoSArray_v {
    float *ff;
    float *ss;
};
void farray_push_ff(Force *ff, FoArray *a) {
    a->ff = (float*) ff;
    a->stress = false;
}

void farray_push_ss(float *ss, FoArray *a) {
    a->ss = ss;
    a->stress = true;
}

bool farray_has_stress(const FoArray *a) {
    return a->stress;
}

void farray_get_view(const FoArray *a, FoArray_v *v) {
    v->ff = a->ff;
}

void farray_get_view(const FoArray *a, FoSArray_v *v) {
    v->ff = a->ff;
    v->ss = a->ss;
}







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
#endif /* UDX_EXCH_TYPE_COMMON */

#ifndef UDX_STRUCT_PARRAY_COMMON
#define UDX_STRUCT_PARRAY_COMMON
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
#ifndef UDX_STRUCT_FARRAY_COMMON
#define UDX_STRUCT_FARRAY_COMMON
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
enum {
    MAX_SIZE = 10
};

struct PFarrays {
    PFarray a[MAX_SIZE];
    int n;
};
void pfarrays_ini(PFarrays **pfa) {
    EMALLOC(1, pfa);
    pfarrays_clear(*pfa);
}

void pfarrays_fin(PFarrays *p) {
    EFREE(p);
}

void pfarrays_clear(PFarrays *p) {
    p->n = 0;
}

void pfarrays_push(PFarrays *pp, PFarray p) {
    int i = pp->n ++;
    if ( pp->n > MAX_SIZE )
        ERR("pushed too many pfarrays: %d/%d", pp->n, MAX_SIZE);
    pp->a[i] = p;
}

void pfarrays_push(PFarrays *pp, long n, PaArray p, FoArray f) {
    PFarray pf;
    pf.p = p;
    pf.n = n;
    pf.f = f;
    UC(pfarrays_push(pp, pf));
}

int pfarrays_size(const PFarrays *p) {
    return p->n;
}

void pfarrays_get(int i, const PFarrays *pf, long *n, PaArray *p, FoArray *f) {
    const PFarray *pfa;

    if (i >= pf->n)
        ERR("Out of bounds: %d/%d", i, pf->n);

    pfa = &pf->a[i];
    *n = pfa->n;
    *p = pfa->p;
    *f = pfa->f;
}

void pfarrays_get(int i, const PFarrays *pp, PFarray *p) {
    UC(pfarrays_get(i, pp, &p->n, &p->p, &p->f));
}

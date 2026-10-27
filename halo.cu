
// Editable compilation unit for particle distribution and fluid halo exchange.


// Grouped editable compilation unit for all distribution units.


// Grouped editable compilation unit for distribution exchange and rigid distribution.


// Grouped editable compilation unit for distribution exchange helpers.


// Grouped editable compilation unit for distribution and exchange common code.


#include "config.cuh"

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


#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#include "device.cuh"
#endif
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
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
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
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
#endif

#ifndef UDX_DISTR_EXCH_DMAP_TYPE_COMMON
#define UDX_DISTR_EXCH_DMAP_TYPE_COMMON
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
#endif

struct Particle;
struct int3;

void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf); // <1>
void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp); // <3>


namespace dcommon_dev {
/* pack packets of nv particles into 27 buffers according to map  */
__global__ void pack_pp_packets(int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf) {
    int i, cid, fid, scid;
    int dst, src, offset;
    i   = threadIdx.x + blockDim.x * blockIdx.x;
    cid = blockIdx.y;

    if (i >= nv) return;
    fid = frag_dev::frag_get_fid(m.starts, cid);

    offset = cid - m.starts[fid];
    scid   = m.ids[fid][offset];

    dst = nv * offset + i;
    src = nv * scid   + i;

    buf.d[fid][dst] = pp[src];
}

static __device__ void fid2shift(int3 L, int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = L.x * frag_dev::i2dx(id);
    s[Y] = L.y * frag_dev::i2dy(id);
    s[Z] = L.z * frag_dev::i2dz(id);
}

static  __device__ void shift_1p(const int s[3], /**/ Particle *p) {
    enum {X, Y, Z};
    p->r[X] += s[X];
    p->r[Y] += s[Y];
    p->r[Z] += s[Z];
}

__global__ void shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    fid2shift(L, fid, /**/ s);
    shift_1p(s, /**/ pp + i);
}

__global__ void shift_halo(int3 L, const Sarray<int, 27> starts, /**/ Particle *pp) {
    int pid, fid, s[3];

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= starts.d[26]) return;
    fid = frag_dev::frag_get_fid(starts.d, pid);

    fid2shift(L, fid, s);
    shift_1p(s, /**/ pp + pid);
}
}

void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf) {
    enum {THR=128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(nv, THR), nc);

    KL(dcommon_dev::pack_pp_packets, (blck, thrd), (nv, pp, m, /**/ buf));
}

void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp) {
    KL(dcommon_dev::shift_one_frag, (k_cnf(n)), (L, n, fid, /**/ pp));
}

void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp) {
    KL(dcommon_dev::shift_halo, (k_cnf(nhalo)), (L, starts, /**/ pp));
}





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
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
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
#endif

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

#endif /* UDX_DISTR_EXCH_COMM_COMMON */
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
namespace exch_dev {
static __device__ void pack_p(const Particle *pp, int offset, int *indices, int cap, int i, /**/ Particle *buf) {
    int dst, src;
    dst = offset + i;
    if (dst >= cap) return;
    src = __ldg(indices + dst);
    buf[dst] = pp[src];
}

__global__ void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf) {
    int gid, fid, frag_i, hi, step;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = ph.starts[26];
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        fid = frag_dev::frag_get_fid(ph.starts, gid);

        /* index in the fragment coordinates */
        frag_i = gid - ph.starts[fid];

        pack_p(pp, ph.offsets[fid], ph.indices[fid], ph.cap[fid], frag_i, /**/ buf.d[fid]);
    }
}

// TODO this is copy/paste from distr/common
static __device__ void fid2shift(int3 L, int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = L.x * frag_dev::i2dx(id);
    s[Y] = L.y * frag_dev::i2dy(id);
    s[Z] = L.z * frag_dev::i2dz(id);
}

static  __device__ void shift_f3(const int s[3], /**/ float *r) {
    enum {X, Y, Z};
    r[X] += s[X];
    r[Y] += s[Y];
    r[Z] += s[Z];
}

__global__ void ecommon_shift_pp_one_frag(int3 L, int n, const int fid, /**/ Particle *pp) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    fid2shift(L, fid, /**/ s);
    shift_f3(s, /**/ pp[i].r);
}

__global__ void ecommon_shift_rrcp_one_frag(int3 L, int n, const int fid, /**/ Positioncp *rr) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    fid2shift(L, fid, /**/ s);
    shift_f3(s, /**/ rr[i].rp);
    shift_f3(s, /**/ rr[i].rc);
}
}

void ecommon_pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf) {
    KL(exch_dev::ecommon_pack_pp, (14 * 16, 128), (pp, ph, /**/ buf));
}

void ecommon_shift_pp_one_frag(int3 L, int n, const int fid, /**/ Particle *pp) {
    KL(exch_dev::ecommon_shift_pp_one_frag, (k_cnf(n)), (L, n, fid, /**/ pp));
}

void ecommon_shift_rrcp_one_frag(int3 L, int n, const int fid, /**/ Positioncp *rr) {
    KL(exch_dev::ecommon_shift_rrcp_one_frag, (k_cnf(n)), (L, n, fid, /**/ rr));
}





// Grouped editable compilation unit for distribution and exchange maps.


// Grouped editable compilation unit for fluid status and distribution maps.


#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "api.cuh"

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
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
enum {frag_bulk = 26};
#endif

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


enum {SUCCESS, PACK_FAILURE};
struct DFluStatus {
    int errorcode;
    int cap, cnt, fid; /* capacity, count, fragment */
};

void dflu_status_ini(DFluStatus **ps) {
    DFluStatus *s;
    EMALLOC(1, &s);
    s->errorcode = SUCCESS;
    *ps = s;
}

void dflu_status_fin(DFluStatus *s) {
    EFREE(s);
}

int  dflu_status_success(DFluStatus *s) {
    return s->errorcode == SUCCESS;
}

static void pack_failure(DFluStatus *s) {
    enum {X, Y, Z};
    int cap, cnt, fid, d[3];
    cap = s->cap; cnt = s->cnt; fid = s->fid;
    d[X] = frag_hst::i2dx(fid);
    d[Y] = frag_hst::i2dy(fid);
    d[Z] = frag_hst::i2dz(fid);
    msg_print("exceed capacity, fragment %d = [%d %d %d]: %d/%d",
              fid, d[X], d[Y], d[Z], cnt, cap);
}
void dflu_status_log(DFluStatus *s) {
    int code;
    if (s == NULL) msg_print("DFluStatus: s == NULL");
    else {
        code = s->errorcode;
        if      (code == SUCCESS)      msg_print("DFluStatus: SUCCESS");
        else if (code == PACK_FAILURE) pack_failure(s);
        else ERR("unknown errorcode = %d\n", code);
    }
}

void dflu_status_exceed(int fid, int cnt, int cap, /**/ DFluStatus *s) {
    if (s == NULL) ERR("status == NULL");
    s->fid = fid; s->cnt = cnt; s->cap = cap;
    s->errorcode = PACK_FAILURE;
}

int dflu_status_nullp(DFluStatus *s) { return s == NULL; }




#include "device.cuh"

#include <stdlib.h>
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
#endif /* UDX_DISTR_EXCH_SHFL_COMMON */

#ifndef UDX_DISTR_EXCH_DMAP_TYPE_COMMON
#define UDX_DISTR_EXCH_DMAP_TYPE_COMMON
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
#endif

void dmap_ini(int nfrags, const int capacity[], /**/ DMap *m) {
    Dalloc(&m->counts,  nfrags);
    Dalloc(&m->starts, (nfrags + 1));

    CC(d::alloc_pinned((void**) &m->hcounts, nfrags * sizeof(int)));

    int i, c;
    for (i = 0; i < nfrags; ++i) {
        c = capacity[i];
        if (c) Dalloc(&m->ids[i], c);
        else   m->ids[i] = NULL;
    }

}

void dmap_fin(int nfrags, /**/ DMap *m) {
    Dfree(m->counts);
    Dfree(m->starts);
    CC(d::FreeHost(m->hcounts));
    for (int i = 0; i < nfrags; ++i)
        if (m->ids[i])
            Dfree(m->ids[i]);
}

void dmap_reini(int nfrags, /**/ DMap m) {
    CC(d::MemsetAsync(m.counts, 0, nfrags * sizeof(int)));
}

void dmap_download_counts(int nfrags, /**/ DMap *m) {
    CC(d::MemcpyAsync(m->hcounts, m->counts, nfrags * sizeof(int), D2H));
}

void dmap_ini_host(int nfrags, const int capacity[], /**/ DMap *m) {
    EMALLOC(nfrags    ,  &m->counts);
    EMALLOC(nfrags + 1,  &m->starts);

    int i, c;
    for (i = 0; i < nfrags; ++i) {
        c = capacity[i];
        if (c)
            EMALLOC(c, &m->ids[i]);
    }
}

void dmap_fin_host(int nfrags, /**/ DMap *m) {
    EFREE(m->counts);
    EFREE(m->starts);
    for (int i = 0; i < nfrags; ++i)
        EFREE(m->ids[i]);
}

void dmap_reini_host(int nfrags, /**/ DMap m) {
    memset(m.counts, 0, nfrags * sizeof(int));
}

void dmap_D2H(int nfrags, const DMap *d, /**/ DMap *h) {
    CC(d::MemcpyAsync(h->counts, d->counts,  nfrags      * sizeof(int), D2H));
    CC(d::MemcpyAsync(h->starts, d->starts, (nfrags + 1) * sizeof(int), D2H));

    dSync();

    int i, c;
    for (i = 0; i < nfrags; ++i) {
        c = h->counts[i];
        if (c)
            CC(d::MemcpyAsync(h->ids[i], d->ids[i], c * sizeof(int), D2H));
    }
}





#include "device.cuh"

#include <stdlib.h>
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

#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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

// [E]xchange map

enum {MAX_FRAGS=26};

struct EMap {
    int *counts;         /* number of entities leaving in each fragment */
    int *starts;         /* cumulative sum of the above                 */
    int *offsets;        /* offsets per fragment for each solute        */
    int *ids[MAX_FRAGS]; /* indices of leaving objects                  */
    int *cap;            /* capacity of ids                             */
};

void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map);               // <1>
void emap_fin(int nfrags, EMap *map);                                       // <2>
void emap_reini(int nw, int nfrags, /**/ EMap map);                         // <3>
void emap_scan(int nw, int nfrags, /**/ EMap map);                          // <4>
void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]); // <5>
void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int tot_counts[], int *counts[]);
#define _I_ static __device__

#if __CUDACC_VER_MAJOR__ >= 9
template <typename T>
_I_ T emap_shfl_down(T val, unsigned int width) {
    return __shfl_down_sync(0xFFFFFFFF, val, width);
}

template <typename T>
_I_ T emap_shfl_up(T val, unsigned int width) {
    return __shfl_up_sync(0xFFFFFFFF, val, width);
}
template <typename T>
_I_ T emap_shfl(T val, unsigned int width) {
    return __shfl_sync(0xFFFFFFFF, val, width);
}
#else
template <typename T>
_I_ T emap_shfl_down(T val, unsigned int width) {
    return __shfl_down(val, width);
}

template <typename T>
_I_ T emap_shfl_up(T val, unsigned int width) {
    return __shfl_up(val, width);
}
template <typename T>
_I_ T emap_shfl(T val, unsigned int width) {
    return __shfl(val, width);
}
#endif

#undef _I_
namespace emap_dev {

static __device__ int warpexscan(int cnt, int t) {
    assert (t < warpSize);
    int L, scan;
    scan = cnt;
    for (L = 1; L < warpSize; L <<= 1) scan += (t >= L) * emap_shfl_up(scan, L);
    return scan - cnt;
}

__global__ void scan2d(int nfrags, const int *counts, const int *offsets, /**/ int *nextoffsets, int *starts) {
    int t, cnt, scan, nextoffset;
    t = threadIdx.x;
    cnt = 0;
    if (t < nfrags) {
        cnt = counts[t];
        nextoffset = cnt + offsets[t];
        nextoffsets[t] = nextoffset;
    }
    scan = warpexscan(cnt, t);
    if (t <= nfrags) starts[t] = scan;
}

} // dev

static void check_frags(int nfrags) {
    if (nfrags > MAX_FRAGS)
        ERR("Too many fragments: %d/%d", nfrags, MAX_FRAGS);
}
static int get_stride(int nfrags)       {return nfrags + 1;}
static int get_size(int nw, int nfrags) {return (nw + 1) * get_stride(nfrags);}

void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map) {
    int i, c, sz;
    UC(check_frags(nfrags));
    sz = get_size(nw, nfrags);
    Dalloc(&map->counts,  sz);
    Dalloc(&map->starts,  sz);
    Dalloc(&map->offsets, sz);
    Dalloc(&map->cap, nfrags);
    cH2D(map->cap, cap, nfrags);

    for (i = 0; i < nfrags; ++i) {
        c = cap[i];
        Dalloc(&map->ids[i], c);
    }
}

void emap_fin(int nfrags, EMap *map) {
    UC(check_frags(nfrags));
    Dfree(map->counts);
    Dfree(map->starts);
    Dfree(map->offsets);
    Dfree(map->cap);

    for (int i = 0; i < nfrags; ++i)
        Dfree(map->ids[i]);
}

void emap_reini(int nw, int nfrags, /**/ EMap map) {
    size_t sz;
    UC(check_frags(nfrags));
    sz = get_size(nw, nfrags) * sizeof(int);
    if (sz == 0) return;
    CC(d::MemsetAsync(map.counts,  0, sz));
    CC(d::MemsetAsync(map.starts,  0, sz));
    CC(d::MemsetAsync(map.offsets, 0, sz));
}

void emap_scan(int nw, int nfrags, /**/ EMap map) {
    int i, *cc, *ss, *oo, *oon, stride;
    UC(check_frags(nfrags));
    stride = get_stride(nfrags);
    for (i = 0; i < nw; ++i) {
        cc  = map.counts  + i * stride;
        ss  = map.starts  + i * stride;
        oo  = map.offsets + i * stride;
        oon = oo + stride; /* oo next */
        KL(emap_dev::scan2d, (1, 32), (nfrags, cc, oo, /**/ oon, ss));
    }
}

void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]) {
    int *src, stride;
    UC(check_frags(nfrags));
    stride = get_stride(nfrags);
    src = map.offsets + nw * stride;
    cD2H(counts, src, nfrags);
}

static void transpose(int nw, int stride, int nfrags, const int *buf, /**/ int tot_counts[], int *counts[]) {
    int i, j, c, tc;
    for (i = 0; i < nfrags; ++i) {
        tc = 0;
        for (j = 0; j < nw; ++j) {
            c = buf[j*stride + i];
            tc += c;
            counts[i][j] = c;
        }
        tot_counts[i] = tc;
    }
}

void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int tot_counts[], int *counts[]) {
    int *buf, stride, sz;
    UC(check_frags(nfrags));
    stride = get_stride(nfrags);
    sz = stride * nw;
    EMALLOC(stride * nw, &buf);
    cD2H(buf, map.counts, sz);
    transpose(nw, stride, nfrags, buf, /**/ tot_counts, counts);
    EFREE(buf);
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


void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#include "device.cuh"
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
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
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif /* UDX_DISTR_EXCH_KL_COMMON */
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
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
#include "types.cuh"
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
#ifndef UDX_DISTR_EXCH_DEF_COMMON
#define UDX_DISTR_EXCH_DEF_COMMON
#include "constants.cuh"
#endif /* UDX_DISTR_EXCH_DEF_COMMON */

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
#endif /* UDX_DISTR_EXCH_FRAG_DEV_COMMON */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_DISTR_EXCH_FRAG_IMP_COMMON */
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

#endif /* UDX_DISTR_EXCH_COMM_COMMON */
#ifndef UDX_DISTR_EXCH_COMM_UTILS_COMMON
#define UDX_DISTR_EXCH_COMM_UTILS_COMMON
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
#endif /* UDX_DISTR_EXCH_COMM_UTILS_COMMON */

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
#endif /* UDX_DISTR_EXCH_SHFL_COMMON */
#ifndef UDX_DISTR_EXCH_DMAP_TYPE_COMMON
#define UDX_DISTR_EXCH_DMAP_TYPE_COMMON
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
#endif /* UDX_DISTR_EXCH_DMAP_TYPE_COMMON */
#ifndef UDX_DISTR_EXCH_DMAP_DEV_COMMON
#define UDX_DISTR_EXCH_DMAP_DEV_COMMON
/* exclusive scan */
template <int NCOUNTS>
static __global__ void dmap_scan(/**/ DMap m)              // <1>
{
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0, cnt = 0;

    if (tid < NCOUNTS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * shfl_up(val, L) ;
    if (tid <= NCOUNTS) m.starts[tid] = val - cnt;
}

static __device__ int dmap_get_fid(int3 L, const float r[3])       // <2>
{
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -L.x/2) + (r[X] >= L.x/2);
    y = -1 + (r[Y] >= -L.y/2) + (r[Y] >= L.y/2);
    z = -1 + (r[Z] >= -L.z/2) + (r[Z] >= L.z/2);
    return frag_dev::d2i(x, y, z);
}

static __device__ void dmap_add(int pid, int fid, DMap m)  // <3>
{
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}
#endif /* UDX_DISTR_EXCH_DMAP_DEV_COMMON */
void dmap_ini(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin(int nfrags, /**/ DMap *m);
void dmap_reini(int nfrags, /**/ DMap m);
void dmap_download_counts(int nfrags, /**/ DMap *m);
void dmap_ini_host(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin_host(int nfrags, /**/ DMap *m);
void dmap_reini_host(int nfrags, /**/ DMap m);
void dmap_D2H(int nfrags, const DMap *d, /**/ DMap *h);
struct Particle;
struct int3;

void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf); // <1>
void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp); // <3>


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

namespace drig_dev {

__global__ void build_map(int3 L, int n, const Rigid *ss, /**/ DMap m) {
    enum {X, Y, Z};
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float r[3];

    r[X] = ss[i].com[X];
    r[Y] = ss[i].com[Y];
    r[Z] = ss[i].com[Z];

    fid = dmap_get_fid(L, r);
    dmap_add(i, fid, /**/ m);
}

__global__ void pack_ss(const Rigid *ss, DMap m, /**/ Sarray<Rigid*, 27> buf) {
    int i, fid; /* [f]ragment [id] */
    int d, s;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    fid = frag_dev::frag_get_fid(m.starts, i);
    if (i >= m.starts[27]) return;

    d = i - m.starts[fid];
    s = __ldg(m.ids[fid] + d);

    buf.d[fid][d] = ss[s];
}

}
/*
 ipp: particles of the mesh
 ss:  "Rigid" structures
*/

enum {
    ID_PP,
    ID_SS,
    DRIG_MAX_NBAGS
};

struct DRigPack {
    DMap map;
    dBags dbags[DRIG_MAX_NBAGS], *dipp, *dss;
    hBags hbags[DRIG_MAX_NBAGS], *hipp, *hss;
    CommBuffer *hbuf;
    int nbags;
    int3 L;  /* subdomain size */
};

struct DRigComm {
    Comm *comm;
};

struct DRigUnpack {
    hBags hbags[DRIG_MAX_NBAGS], *hipp, *hss;
    CommBuffer *hbuf;
    int nbags;
    int3 L;  /* subdomain size */
};
static void drig_get_num_capacity(int maxns, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxns;
}

void drig_pack_ini(int3 L, int maxns, int nv, DRigPack **pack) {
    int numc[NBAGS];
    DRigPack *p;
    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    drig_get_num_capacity(maxns, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    p->hipp = &p->hbags[ID_PP];  p->hss = &p->hbags[ID_SS];
    p->dipp = &p->dbags[ID_PP];  p->dss = &p->dbags[ID_SS];

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ p->hipp, p->dipp));
    UC(comm_bags_ini(PINNED, DEV_ONLY, sizeof(Rigid), numc, /**/ p->hss, p->dss));

    p->nbags = DRIG_MAX_NBAGS;

    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void drig_comm_ini(MPI_Comm cart, /**/ DRigComm **com) {
    DRigComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void drig_unpack_ini(int3 L, int maxns, int nv, DRigUnpack **unpack) {
    int numc[NBAGS];
    DRigUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    drig_get_num_capacity(maxns, /**/ numc);

    u->hipp = &u->hbags[ID_PP];  u->hss = &u->hbags[ID_SS];

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ u->hipp, NULL));

    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Rigid), numc, /**/ u->hss, NULL));

    u->nbags = DRIG_MAX_NBAGS;

    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}
void drig_pack_fin(DRigPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ p->hipp, p->dipp));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ p->hss, p->dss));
    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void drig_comm_fin(DRigComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void drig_unpack_fin(DRigUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, u->hipp, NULL));
    UC(comm_bags_fin(HST_ONLY, NONE, u->hss, NULL));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}
static void build_map(int3 L, int n, const Rigid *ss, /**/ DMap m) {
    UC(dmap_reini(NBAGS, /**/ m));
    KL(drig_dev::build_map, (k_cnf(n)), (L, n, ss, /**/ m));
    KL(dmap_scan<NBAGS>, (1, 32), (/**/ m));
}

void drig_build_map(int ns, const Rigid *ss, /**/ DRigPack *p) {
    UC(build_map(p->L, ns, ss, /**/ p->map));
}
static void drig_pack_pp(const DMap m, int ns, int nv, const Particle *ipp, /**/ dBags *bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(*bags, &wrap);

    dcommon_pack_pp_packets(ns, nv, ipp, m, /**/ wrap);
}

static void pack_ss(const DMap m, int n, const Rigid *ss, /**/ dBags *bags) {

    Sarray<Rigid*, 27> wrap;
    bag2Sarray(*bags, &wrap);

    KL((drig_dev::pack_ss), (k_cnf(n)), (ss, m, /**/ wrap));
}


void drig_pack(int ns, int nv, const Rigid *ss, const Particle *ipp, /**/ DRigPack *p) {
    UC(drig_pack_pp(p->map, ns, nv, ipp, /**/ p->dipp));
    UC(pack_ss(p->map, ns, ss, /**/ p->dss));
}

void drig_download(DRigPack *p) {
    CC(d::Memcpy(p->hipp->counts, p->map.counts, NBAGS * sizeof(int), D2H));
    CC(d::Memcpy(p->hss->counts, p->map.counts, NBAGS * sizeof(int), D2H));
}
void drig_post_recv(DRigComm *c, DRigUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void drig_post_send(DRigPack *p, DRigComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void drig_wait_recv(DRigComm *c, DRigUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void drig_wait_send(DRigComm *c) {
    UC(comm_wait_send(c->comm));
}
void drig_unpack_bulk(const DRigPack *p, /**/ RigQuants *q) {
    int ns, nv, n;
    ns = p->hipp->counts[frag_bulk];
    nv = q->nv;
    n = ns * nv;

    if (ns) {
        CC(d::MemcpyAsync(q->i_pp, p->dipp->data[frag_bulk], n * sizeof(Particle), D2D));
        CC(d::MemcpyAsync(  q->ss,  p->dss->data[frag_bulk],   ns * sizeof(Rigid), D2D));
    }
    q->ns = ns;
}

static void shift(int3 L, int fid, float r[3]) {
    enum {X, Y, Z};
    r[X] += L.x * frag_hst::i2dx(fid);
    r[Y] += L.y * frag_hst::i2dy(fid);
    r[Z] += L.z * frag_hst::i2dz(fid);
}

static void shift_ss_one_frag(int3 L, int n, int fid, Rigid *ss) {
    for (int i = 0; i < n; ++i) shift(L, fid, ss[i].com);
}

void drig_unpack_halo(const DRigUnpack *u, /**/ RigQuants *q) {
    int ns, nv, n, i, strtp, strts;
    size_t szp, szs;
    nv = q->nv;
    strts = q->ns;
    strtp = strts * nv;

    for (i = 0; i < NFRAGS; ++i) {
        ns = u->hss->counts[i];
        n  = ns * nv;
        szp = n * sizeof(Particle);
        szs = ns * sizeof(Rigid);

        if (ns) {
            /* particles */
            CC(d::MemcpyAsync(q->i_pp + strtp, u->hipp->data[i], szp, H2D));
            dcommon_shift_one_frag(u->L, n, i, /**/ q->i_pp + strtp);
            /* solid */
            shift_ss_one_frag(u->L, ns, i, /**/ (Rigid*) u->hss->data[i]);
            CC(d::MemcpyAsync(q->ss + strts, u->hss->data[i], szs, H2D));
        }
        strtp += n;
        strts += ns;
    }
    q->ns = strts;
}






// Grouped editable compilation unit for fluid and RBC distribution.


#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "api.cuh"

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
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif /* UDX_DISTR_EXCH_KL_COMMON */
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

#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#include "device.cuh"
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
#include "types.cuh"
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
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
#endif /* UDX_DISTR_EXCH_FRAG_DEV_COMMON */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_DISTR_EXCH_FRAG_IMP_COMMON */

#ifndef UDX_DISTR_PARTLIST_TYPE_COMMON
#define UDX_DISTR_PARTLIST_TYPE_COMMON
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
#endif /* UDX_DISTR_PARTLIST_TYPE_COMMON */
static __device__ bool is_dead(int i, const PartList lp) {
    if (lp.deathlist)
        return lp.deathlist[i] == DEAD;
    return false;
}

struct Scan;

void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
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
#endif /* UDX_DISTR_EXCH_SHFL_COMMON */

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
#ifndef UDX_DISTR_FLU_QUANTS_COMMON
#define UDX_DISTR_FLU_QUANTS_COMMON
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
#endif /* UDX_DISTR_FLU_QUANTS_COMMON */

struct Coords;
struct GenColor;

void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q);
void flu_fin(FluQuants *q);

void flu_gen_quants(const Coords*, int numdensity, const GenColor *gc, FluQuants *q); // <1>
void flu_gen_ids(MPI_Comm, const int n, FluQuants *q); // <2>

void flu_strt_quants(MPI_Comm, const char *base, const int id, FluQuants *q);     // <1>
void flu_strt_dump(MPI_Comm, const char *base, const int id, const FluQuants *q); // <2>

void flu_txt_dump(const Coords*, const FluQuants *q); // <1>

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void flu_build_cells(/**/ FluQuants *q); // <2>

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

#endif /* UDX_DISTR_EXCH_COMM_COMMON */
#ifndef UDX_DISTR_EXCH_COMM_UTILS_COMMON
#define UDX_DISTR_EXCH_COMM_UTILS_COMMON
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
#endif /* UDX_DISTR_EXCH_COMM_UTILS_COMMON */

#ifndef UDX_DISTR_EXCH_DMAP_TYPE_COMMON
#define UDX_DISTR_EXCH_DMAP_TYPE_COMMON
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
#endif /* UDX_DISTR_EXCH_DMAP_TYPE_COMMON */
struct DFluPack;
struct DFluComm;
struct DFluUnpack;
struct DFluStatus;

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


/* cell lists */
void dflu_bulk(PartList lp, /**/ FluQuants *q);
void dflu_halo(const DFluUnpack *u, /**/ FluQuants *q);
void dflu_gather(int ndead, const DFluPack *p, const DFluUnpack *u, /**/ FluQuants *q);

#ifndef UDX_DISTR_EXCH_DMAP_DEV_COMMON
#define UDX_DISTR_EXCH_DMAP_DEV_COMMON
/* exclusive scan */
template <int NCOUNTS>
static __global__ void dmap_scan(/**/ DMap m)              // <1>
{
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0, cnt = 0;

    if (tid < NCOUNTS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * shfl_up(val, L) ;
    if (tid <= NCOUNTS) m.starts[tid] = val - cnt;
}

static __device__ int dmap_get_fid(int3 L, const float r[3])       // <2>
{
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -L.x/2) + (r[X] >= L.x/2);
    y = -1 + (r[Y] >= -L.y/2) + (r[Y] >= L.y/2);
    z = -1 + (r[Z] >= -L.z/2) + (r[Z] >= L.z/2);
    return frag_dev::d2i(x, y, z);
}

static __device__ void dmap_add(int pid, int fid, DMap m)  // <3>
{
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}
#endif /* UDX_DISTR_EXCH_DMAP_DEV_COMMON */
void dmap_ini(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin(int nfrags, /**/ DMap *m);
void dmap_reini(int nfrags, /**/ DMap m);
void dmap_download_counts(int nfrags, /**/ DMap *m);
void dmap_ini_host(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin_host(int nfrags, /**/ DMap *m);
void dmap_reini_host(int nfrags, /**/ DMap m);
void dmap_D2H(int nfrags, const DMap *d, /**/ DMap *h);
struct Particle;
struct int3;

void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf); // <1>
void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp); // <3>

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


typedef Sarray <float2*, 26> float2p26;
typedef Sarray <int*, 26> intp26;

enum {
    DFLU_MAX_NBAGS = 3
};

struct DFluPack {
    DMap map;
    dBags dbags[DFLU_MAX_NBAGS], *dpp, *dii, *dcc;
    hBags hbags[DFLU_MAX_NBAGS], *hpp, *hii, *hcc;
    int nbags;
    int nhalo; /* number of sent particles */
    int3 L; /* subdomain size */
    CommBuffer *hbuf;
};

struct DFluComm {
    Comm *comm;
};

struct DFluUnpack {
    hBags hbags[DFLU_MAX_NBAGS], *hpp, *hii, *hcc;
    int nbags;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo; /* number of received particles */
    int3 L; /* subdomain size */
    CommBuffer *hbuf;
};

namespace dflu_dev {
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
}
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


void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
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
#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#include "device.cuh"
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
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
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif /* UDX_DISTR_EXCH_KL_COMMON */
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
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
#include "types.cuh"
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
#ifndef UDX_DISTR_EXCH_DEF_COMMON
#define UDX_DISTR_EXCH_DEF_COMMON
#include "constants.cuh"
#endif /* UDX_DISTR_EXCH_DEF_COMMON */

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
#endif /* UDX_DISTR_EXCH_FRAG_DEV_COMMON */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_DISTR_EXCH_FRAG_IMP_COMMON */
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

#endif /* UDX_DISTR_EXCH_COMM_COMMON */
#ifndef UDX_DISTR_EXCH_COMM_UTILS_COMMON
#define UDX_DISTR_EXCH_COMM_UTILS_COMMON
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
#endif /* UDX_DISTR_EXCH_COMM_UTILS_COMMON */

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
#endif /* UDX_DISTR_EXCH_SHFL_COMMON */
struct Particle;
struct float3;
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);

#ifndef UDX_DISTR_EXCH_DMAP_TYPE_COMMON
#define UDX_DISTR_EXCH_DMAP_TYPE_COMMON
/* [D]istr Map */
struct DMap {
    int *counts;  /* number of entities leaving in each fragment */
    int *starts;  /* cumulative sum of the above                 */
    int *ids[27]; /* indices of leaving objects                  */

    int *hcounts; /* counts on host (pinned) */
};
#endif /* UDX_DISTR_EXCH_DMAP_TYPE_COMMON */
#ifndef UDX_DISTR_EXCH_DMAP_DEV_COMMON
#define UDX_DISTR_EXCH_DMAP_DEV_COMMON
/* exclusive scan */
template <int NCOUNTS>
static __global__ void dmap_scan(/**/ DMap m)              // <1>
{
    int tid, val, cnt;
    tid = threadIdx.x;
    val = 0, cnt = 0;

    if (tid < NCOUNTS) cnt = val = m.counts[tid];
    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * shfl_up(val, L) ;
    if (tid <= NCOUNTS) m.starts[tid] = val - cnt;
}

static __device__ int dmap_get_fid(int3 L, const float r[3])       // <2>
{
    enum {X, Y, Z};
    int x, y, z;
    x = -1 + (r[X] >= -L.x/2) + (r[X] >= L.x/2);
    y = -1 + (r[Y] >= -L.y/2) + (r[Y] >= L.y/2);
    z = -1 + (r[Z] >= -L.z/2) + (r[Z] >= L.z/2);
    return frag_dev::d2i(x, y, z);
}

static __device__ void dmap_add(int pid, int fid, DMap m)  // <3>
{
    int entry;
    entry = atomicAdd(m.counts + fid, 1);
    m.ids[fid][entry] = pid;
}
#endif /* UDX_DISTR_EXCH_DMAP_DEV_COMMON */
void dmap_ini(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin(int nfrags, /**/ DMap *m);
void dmap_reini(int nfrags, /**/ DMap m);
void dmap_download_counts(int nfrags, /**/ DMap *m);
void dmap_ini_host(int nfrags, const int capacity[], /**/ DMap *m);
void dmap_fin_host(int nfrags, /**/ DMap *m);
void dmap_reini_host(int nfrags, /**/ DMap m);
void dmap_D2H(int nfrags, const DMap *d, /**/ DMap *h);
struct Particle;
struct int3;

void dcommon_pack_pp_packets(int nc, int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf); // <1>
void dcommon_shift_one_frag(int3 L, int n, const int fid, /**/ Particle *pp); // <2>
void dcommon_shift_halo(int3 L, int nhalo, const Sarray<int, 27> starts, /**/ Particle *pp); // <3>


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

namespace drbc_dev {

__global__ void build_map(int3 L, int n, const float3 *minext, const float3 *maxext, /**/ DMap m) {
    enum {X, Y, Z};
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float r[3];
    float3 mine, maxe;

    mine = minext[i];
    maxe = maxext[i];

    r[X] = 0.5f * (mine.x + maxe.x);
    r[Y] = 0.5f * (mine.y + maxe.y);
    r[Z] = 0.5f * (mine.z + maxe.z);

    fid = dmap_get_fid(L, r);
    dmap_add(i, fid, /**/ m);
}

}
enum {
    MAX_NHBAGS = 2,
    MAX_NDBAGS = 1
};

struct DRbcPack {
    DMap map;
    float3 *minext, *maxext;
    dBags dbags[MAX_NDBAGS], *dpp;
    hBags hbags[MAX_NHBAGS], *hpp, *hii;
    int nbags;
    CommBuffer *hbuf;
    DMap hmap; /* host map for ids */

    int3 L; /* subdomain size */
};

struct DRbcComm {
    Comm *comm;
};

struct DRbcUnpack {
    hBags hbags[MAX_NHBAGS], *hpp, *hii;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};
static void get_num_capacity(int maxnc, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxnc;
}

void drbc_pack_ini(bool ids, int3 L, int maxnc, int nv, DRbcPack **pack) {
    int i, numc[NBAGS];
    DRbcPack *p;
    EMALLOC(1, pack);
    p = *pack;

    p->L= L;

    get_num_capacity(maxnc, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    i = 0;
    p->hpp = &p->hbags[i];
    p->dpp = &p->dbags[i++];
    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ p->hpp, p->dpp));

    Dalloc(&p->minext, maxnc);
    Dalloc(&p->maxext, maxnc);

    p->hii = NULL;
    if (ids) {
        p->hii = &p->hbags[i++];
        UC(dmap_ini_host(NBAGS, numc, /**/ &p->hmap));
        UC(comm_bags_ini(HST_ONLY, HST_ONLY, sizeof(int), numc, /**/ p->hii, NULL));
    }
    p->nbags = i;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void drbc_comm_ini(MPI_Comm cart, /**/ DRbcComm **com) {
    DRbcComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void drbc_unpack_ini(bool ids, int3 L, int maxnc, int nv, DRbcUnpack **unpack) {
    int i, numc[NBAGS];
    DRbcUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;

    get_num_capacity(maxnc, /**/ numc);

    i = 0;
    u->hpp = &u->hbags[i++];
    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ u->hpp, NULL));

    u->hii = NULL;
    if (ids) {
        u->hii = &u->hbags[i++];
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), numc, /**/ u->hii, NULL));
    }
    u->nbags = i;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}
void drbc_pack_fin(DRbcPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ p->hpp, p->dpp));
    Dfree(p->minext);
    Dfree(p->maxext);

    if (p->hii) {
        UC(dmap_fin_host(NBAGS, /**/ &p->hmap));
        UC(comm_bags_fin(HST_ONLY, HST_ONLY, /**/ p->hii, NULL));
    }
    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void drbc_comm_fin(DRbcComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void drbc_unpack_fin(DRbcUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, /**/ u->hpp, NULL));
    if (u->hii)
        UC(comm_bags_fin(HST_ONLY, NONE, /**/ u->hii, NULL));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}
static void build_map(int3 L, int n, const float3 *minext, const float3 *maxext, /**/ DMap m) {
    UC(dmap_reini(NBAGS, /**/ m));
    KL(drbc_dev::build_map, (k_cnf(n)), (L, n, minext, maxext, /**/ m));
    KL(dmap_scan<NBAGS>, (1, 32), (/**/ m));
}

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p) {
    minmax(pp, nv, nc, /**/ p->minext, p->maxext);
    build_map(p->L, nc, p->minext, p->maxext, /**/  p->map);
}
static void drbc_pack_pp(const DMap m, int nc, int nv, const Particle *pp, /**/ dBags *bags) {
    Sarray<Particle*, NBAGS> wrap;
    bag2Sarray(*bags, &wrap);
    UC(dcommon_pack_pp_packets(nc, nv, pp, m, /**/ wrap));
}

/* all data (including map) on host */
static void pack_ii(const DMap m, int, const int *ii, /**/ hBags *bags) {
    Sarray<int*, NBAGS> wrap;
    bag2Sarray(*bags, &wrap);

    int i, c, j, src;
    for (i = 0; i < NBAGS; ++i) {
        c = m.counts[i];
        for (j = 0; j < c; ++j) {
            src = m.ids[i][j];
            wrap.d[i][j] = ii[src];
        }
    }
}

void drbc_pack(const RbcQuants *q, /**/ DRbcPack *p) {
    UC(drbc_pack_pp(p->map, q->nc, q->nv, q->pp, /**/ p->dpp));
    if (p->hii) {
        UC(dmap_D2H(NBAGS, &p->map, /**/ &p->hmap));
        dSync();
        UC(pack_ii(p->hmap, q->nc, q->ii, /**/ p->hii));
    }
}

void drbc_download(DRbcPack *p) {
    size_t sz = NBAGS * sizeof(int);
    CC(d::Memcpy(p->hpp->counts, p->map.counts, sz, D2H));
    if (p->hii)
        memcpy(p->hii->counts, p->hmap.counts, sz);
}
void drbc_post_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(comm_post_recv(u->hbuf, c->comm));
}

void drbc_post_send(DRbcPack *p, DRbcComm *c) {
    UC(comm_buffer_set(p->nbags, p->hbags, p->hbuf));
    UC(comm_post_send(p->hbuf, c->comm));
}

void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u) {
    UC(comm_wait_recv(c->comm, u->hbuf));
    UC(comm_buffer_get(u->hbuf, u->nbags, u->hbags));
}

void drbc_wait_send(DRbcComm *c) {
    UC(comm_wait_send(c->comm));
}
static int unpack_bulk_pp(const DRbcPack *p, /**/ Particle *pp) {
    int nc = p->hpp->counts[frag_bulk];
    void *src = p->dpp->data[frag_bulk];
    size_t sz = nc * p->hpp->bsize;

    if (nc)
        CC(d::MemcpyAsync(pp, src, sz, D2D));
    return nc;
}

static void unpack_bulk_ii(const DRbcPack *p, /**/ int *ii) {
    int nc = p->hii->counts[frag_bulk];
    void *src = p->hii->data[frag_bulk];
    memcpy(ii, src, nc * sizeof(int));
}

void drbc_unpack_bulk(const DRbcPack *p, /**/ RbcQuants *q) {
    int nc, nv, n;
    nv = q->nv;

    nc = unpack_bulk_pp(p, /**/ q->pp);
    n = nc * nv;

    if (p->hii)
        unpack_bulk_ii(p, /**/ q->ii);

    q->nc = nc;
    q->n = n;
}

static int unpack_halo_pp(int3 L, int nc0, int nv, const hBags *hpp, /**/ Particle *pp) {
    void *src;
    Particle *dst;
    int i, s, c, n;
    size_t sz, bsz;
    bsz = hpp->bsize;

    s = nc0;
    for (i = 0; i < NFRAGS; ++i) {
        c   = hpp->counts[i];
        src = hpp->data[i];
        dst = pp + s * nv;

        if (c) {
            n = c * nv;
            sz = c * bsz;
            CC(d::MemcpyAsync(dst, src, sz, H2D));
            dcommon_shift_one_frag(L, n, i, /**/ dst);
        }
        s += c;
    }
    return s;
}

static void unpack_halo_ii(int nc0, const hBags *hii, /**/ int *ii) {
    void *src;
    int i, s, c;

    s = nc0;
    for (i = 0; i < NFRAGS; ++i) {
        c   = hii->counts[i];
        src = hii->data[i];
        memcpy(ii + s, src, c * sizeof(int));
        s += c;
    }
}

void drbc_unpack_halo(const DRbcUnpack *u, /**/ RbcQuants *q) {
    int nc0, nc, nv;

    nc0 = q->nc;
    nv  = q->nv;

    nc = unpack_halo_pp(u->L, nc0, nv, u->hpp, /**/ q->pp);

    if (u->hii)
        unpack_halo_ii(nc0, u->hii, /**/ q->ii);

    q->n = nc * nv;
    q->nc = nc;
}






// Grouped editable compilation unit for fluid exchange and halo forces.



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


void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
#include "types.cuh"
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
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
#endif /* UDX_DISTR_EXCH_FRAG_DEV_COMMON */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_DISTR_EXCH_FRAG_IMP_COMMON */

#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#include "device.cuh"
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
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
#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif /* UDX_DISTR_EXCH_KL_COMMON */

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

#endif /* UDX_DISTR_EXCH_COMM_COMMON */
#ifndef UDX_DISTR_EXCH_COMM_UTILS_COMMON
#define UDX_DISTR_EXCH_COMM_UTILS_COMMON
template <typename BAGT, typename T, int N>
static void bag2Sarray(BAGT bags, Sarray<T*, N> *buf)
{
    for (int i = 0; i < N; ++i)
        buf->d[i] = (T*) bags.data[i];
}
#endif /* UDX_DISTR_EXCH_COMM_UTILS_COMMON */

#ifndef UDX_FLU_PARRAY_COMMON
#define UDX_FLU_PARRAY_COMMON
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
#endif /* UDX_FLU_PARRAY_COMMON */
#ifndef UDX_FLU_TYPE_COMMON
#define UDX_FLU_TYPE_COMMON
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
#endif /* UDX_FLU_TYPE_COMMON */
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
#endif /* UDX_DISTR_EXCH_SHFL_COMMON */

struct EFluPack;
struct EFluComm;
struct EFluUnpack;
struct PaArray;

void eflu_pack_ini(bool colors, int3 L, int maxd, EFluPack **p);
void eflu_comm_ini(MPI_Comm comm, /**/ EFluComm **c);
void eflu_unpack_ini(bool colors, int3 L, int maxd, EFluUnpack **u);

void eflu_pack_fin(EFluPack *p);
void eflu_comm_fin(EFluComm *c);
void eflu_unpack_fin(EFluUnpack *u);

void eflu_compute_map(const int *start, const int *count, /**/ EFluPack *p);
void eflu_download_cell_starts(/**/ EFluPack *p);

void eflu_pack(const PaArray *parray, /**/ EFluPack *p);
void eflu_download_data(EFluPack *p);

void eflu_post_recv(EFluComm *c, EFluUnpack *u);
void eflu_post_send(EFluPack *p, EFluComm *c);
void eflu_wait_recv(EFluComm *c, EFluUnpack *u);
void eflu_wait_send(EFluComm *c);

void eflu_unpack(EFluUnpack *u);

using flu::LFrag26;
using flu::RFrag26;

void eflu_get_local_frags(const EFluPack *p, /**/ LFrag26 *lfrags);
void eflu_get_remote_frags(const EFluUnpack *u, /**/ RFrag26 *rfrags);


namespace eflu_dev {
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

} // dev

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




// Grouped editable compilation unit for fluid state and halo forces.



#include <stdio.h>
#include <stdlib.h>
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


void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
#ifndef UDX_DISTR_EXCH_DEF_COMMON
#define UDX_DISTR_EXCH_DEF_COMMON
#ifndef UDX_FLU_HALO_DEF_COMMON
#define UDX_FLU_HALO_DEF_COMMON
#include "constants.cuh"
#endif /* UDX_DISTR_EXCH_DEF_COMMON */
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
#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
#ifndef UDX_FLU_HALO_TYPE_COMMON
#define UDX_FLU_HALO_TYPE_COMMON
#include "types.cuh"
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
#endif

#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#ifndef UDX_FLU_HALO_DEVICE_API_COMMON
#define UDX_FLU_HALO_DEVICE_API_COMMON
#include "device.cuh"
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
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

#ifndef UDX_DISTR_PARTLIST_TYPE_COMMON
#define UDX_DISTR_PARTLIST_TYPE_COMMON
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
#endif /* UDX_DISTR_PARTLIST_TYPE_COMMON */
struct Scan;

void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
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
struct GenColor;
struct Particle;
struct Coords;
struct Config;

void inter_color_ini(GenColor**);
void inter_color_fin(GenColor*);

void inter_color_set_drop(float R, GenColor*); // <1>
void inter_color_set_uniform(GenColor*); // <2>

void inter_color_set_conf(const Config*, GenColor*);

void inter_color_apply_hst(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);
void inter_color_apply_dev(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);


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

#ifndef UDX_DISTR_FLU_QUANTS_COMMON
#define UDX_DISTR_FLU_QUANTS_COMMON
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
#endif /* UDX_DISTR_FLU_QUANTS_COMMON */

struct Coords;
struct GenColor;

void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q);
void flu_fin(FluQuants *q);

void flu_gen_quants(const Coords*, int numdensity, const GenColor *gc, FluQuants *q); // <1>
void flu_gen_ids(MPI_Comm, const int n, FluQuants *q); // <2>

void flu_strt_quants(MPI_Comm, const char *base, const int id, FluQuants *q);     // <1>
void flu_strt_dump(MPI_Comm, const char *base, const int id, const FluQuants *q); // <2>

void flu_txt_dump(const Coords*, const FluQuants *q); // <1>

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void flu_build_cells(/**/ FluQuants *q); // <2>

static void ini_ii(int maxp, int **ii, int **ii0, int **ii_hst) {
    Dalloc(ii, maxp);
    Dalloc(ii0, maxp);
    EMALLOC(maxp, ii_hst);
}

void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q) {
    q->n = 0;
    q->maxp = maxp;
    Dalloc(&q->pp, maxp);
    Dalloc(&q->pp0, maxp);
    UC(clist_ini(L.x, L.y, L.z, /**/ &q->cells));
    UC(clist_ini_map(maxp, 2, q->cells, /**/ &q->mcells));

    EMALLOC(maxp, &q->pp_hst);

    if (ids)    ini_ii(maxp, &q->ii, &q->ii0, &q->ii_hst);
    if (colors) ini_ii(maxp, &q->cc, &q->cc0, &q->cc_hst);

    q->ids = ids;
    q->colors = colors;
}
static void fin_ii(int *ii, int *ii0, int *ii_hst) {
    CC(d::Free(ii));
    CC(d::Free(ii0));
    EFREE(ii_hst);
}

void flu_fin(FluQuants *q) {
    CC(d::Free(q->pp));
    CC(d::Free(q->pp0));
    UC(clist_fin(q->cells));
    UC(clist_fin_map(q->mcells));
    EFREE(q->pp_hst);

    if (q->ids)    UC(fin_ii(q->ii, q->ii0, q->ii_hst));
    if (q->colors) UC(fin_ii(q->cc, q->cc0, q->cc_hst));
}
static void assert_np(int n, int m) {
    if (n > m) ERR("too many particles: n = %d < m = %d", n, m);
}

static int gen0(int3 L, int numdensity, int maxp, Particle *pp) { /* generate particle positions and velocities */
    enum {X, Y, Z};
    UC(assert_np(L.x * L.y * L.z * numdensity, maxp));
    os_srand(123456);
    int iz, iy, ix, l, nd = numdensity;
    int n = 0; /* particle index */
    float x, y, z, dr = 0.99;
    for (iz = 0; iz < L.z; iz++)
    for (iy = 0; iy < L.y; iy++)
    for (ix = 0; ix < L.x; ix++) {
        /* edge of a cell */
        int xlo = -0.5*L.x + ix, ylo = -0.5*L.y + iy, zlo = -0.5*L.z + iz;
        for (l = 0; l < nd; l++) {
            Particle p;
            x = xlo + dr * os_drand(), y = ylo + dr * os_drand(), z = zlo + dr * os_drand();
            p.r[X] = x; p.r[Y] = y; p.r[Z] = z;
            p.v[X] = 0; p.v[Y] = 0; p.v[Z] = 0;
            pp[n++] = p;
        }
    }
    msg_print("ic::gen: %d solvent particles", n);
    return n;
}

static int genColor(int3 L, int numdensity, int maxp, const Coords *coords, const GenColor *gc, /*o*/ Particle *pp, int *color, /*w*/ Particle *pp_hst, int *color_hst) {
    int n = gen0(L, numdensity, maxp, pp_hst);
    inter_color_apply_hst(coords, gc, n, pp_hst, /**/ color_hst);
    cH2D(color, color_hst, n);
    cH2D(   pp,    pp_hst, n);
    return n;
}

static int genGrey(int3 L, int numdensity, int maxp, /*o*/ Particle *dev, /*w*/ Particle *hst) {
    int n = gen0(L, numdensity, maxp, hst);
    cH2D(dev, hst, n);
    return n;
}

void flu_gen_quants(const Coords *coords, int numdensity, const GenColor *gc, FluQuants *q) {
    int3 L = subdomain(coords);
    if (q->colors)
        q->n = genColor(L, numdensity, q->maxp, coords, gc, q->pp, q->cc, /*w*/ q->pp_hst, q->cc_hst);
    else
        q->n = genGrey(L, numdensity, q->maxp, q->pp, /*w*/ q->pp_hst);
}

static void ii_gen0(MPI_Comm comm, const long n, int *ii) {
    long i0 = 0;
    MC(m::Exscan(&n, &i0, 1, MPI_LONG, MPI_SUM, comm));
    for (long i = 0; i < n; ++i) ii[i] = i + i0;
}

static void ii_gen(MPI_Comm comm, const int n, int *ii_dev, int *ii_hst) {
    ii_gen0(comm, n, ii_hst);
    cH2D(ii_dev, ii_hst, n);
}

void flu_gen_ids(MPI_Comm comm, const int n, FluQuants *q) {
    ii_gen(comm, n, q->ii, q->ii_hst);
}
#define CODE "flu"
#define PP  CODE ".pp"
#define COL CODE ".cc"
#define IDS CODE ".ii"

static int strt_pp(MPI_Comm comm, const char *base, const int id, Particle *dev, /*w*/ Particle *hst) {
    int n;
    restart_read_pp(comm, base, PP, id, &n, hst);
    if (n) cH2D(dev, hst, n);
    return n;
}

static int strt_ii(MPI_Comm comm, const char *base, const char *code, const int id, int *dev, /*w*/ int *hst) {
    int n;
    restart_read_ii(comm, base, code, id, &n, hst);
    if (n) cH2D(dev, hst, n);
    return n;
}

void flu_strt_quants(MPI_Comm comm, const char *base, const int id, FluQuants *q) {
    q->n =         strt_pp(comm, base,     id, /**/ q->pp, /* w */ q->pp_hst);
    if (q->ids)    strt_ii(comm, base, IDS, id, /**/ q->ii, /* w */ q->ii_hst);
    if (q->colors) strt_ii(comm, base, COL, id, /**/ q->cc, /* w */ q->cc_hst);
}

static void strt_dump_pp(MPI_Comm comm, const char *base, const int id, const int n, const Particle *dev, Particle *hst) {
    if (n) cD2H(hst, dev, n);
    restart_write_pp(comm, base, PP, id, n, hst);
}

static void strt_dump_ii(MPI_Comm comm, const char *base, const char *code, const int id, const int n, const int *dev, int *hst) {
    if (n) cD2H(hst, dev, n);
    restart_write_ii(comm, base, code, id, n, hst);
}

void flu_strt_dump(MPI_Comm comm, const char *base, const int id, const FluQuants *q) {
    strt_dump_pp(comm, base, id, q->n, q->pp, /* w */ q->pp_hst);
    if (q->ids)    strt_dump_ii(comm, base, IDS, id, q->n, q->ii, /* w */ q->ii_hst);
    if (q->colors) strt_dump_ii(comm, base, COL, id, q->n, q->cc, /* w */ q->cc_hst);
}

#undef CODE
#undef COl
#undef IDS
#undef PP
static void local2global_p(const Coords *c, long n, Particle *pp) {
    enum {X, Y, Z};
    float *r;
    int i;
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        r[X] = xl2xg(c, r[X]);
        r[Y] = yl2yg(c, r[Y]);
        r[Z] = zl2zg(c, r[Z]);
    }
}

static void gen_name(const Coords *c, /**/ char *name) {
    int r;
    char stamp[FILENAME_MAX];
    coord_stamp(c, /**/ stamp);
    r = sprintf(name, "%s.txt", stamp);
    if (r < 0) ERR("sprintf failed");
}

void flu_txt_dump(const Coords *c, const FluQuants *q) {
    Particle *dev, *hst;
    int n;
    char name[FILENAME_MAX];
    n = q->n; dev = q->pp; hst = q->pp_hst;
    if (n) cD2H(hst, dev, n);
    local2global_p(c, n, /**/ hst);
    UC(gen_name(c, /**/ name));
    UC(txt_write_pp(n, hst, name));
}
void flu_build_cells(/**/ FluQuants *q) {
    UC(clist_build(q->n, q->n, q->pp, /**/ q->pp0, q->cells, q->mcells));
    // swap
    Particle *tmp = q->pp;
    q->pp = q->pp0; q->pp0 = tmp;
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
#ifndef UDX_DISTR_EXCH_DEVICE_API_COMMON
#define UDX_DISTR_EXCH_DEVICE_API_COMMON
#ifndef UDX_FLU_HALO_DEVICE_API_COMMON
#define UDX_FLU_HALO_DEVICE_API_COMMON
#include "device.cuh"
#endif /* UDX_DISTR_EXCH_DEVICE_API_COMMON */
#endif

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
#endif /* UDX_DISTR_EXCH_FRAG_DEV_COMMON */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
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
#endif /* UDX_DISTR_EXCH_FRAG_IMP_COMMON */

#ifndef UDX_DISTR_EXCH_DEF_COMMON
#define UDX_DISTR_EXCH_DEF_COMMON
#ifndef UDX_FLU_HALO_DEF_COMMON
#define UDX_FLU_HALO_DEF_COMMON
#include "constants.cuh"
#endif /* UDX_DISTR_EXCH_DEF_COMMON */
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

#ifndef UDX_DISTR_EXCH_TYPE_COMMON
#define UDX_DISTR_EXCH_TYPE_COMMON
#ifndef UDX_FLU_HALO_TYPE_COMMON
#define UDX_FLU_HALO_TYPE_COMMON
#include "types.cuh"
#endif /* UDX_DISTR_EXCH_TYPE_COMMON */
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
/* maximum of pairwise dpd parameters */
enum {
    MAX_PAR = (N_COLOR * (N_COLOR + 1)) / 2
};


struct PairDPD {
    float a, g, s, spow;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], spow;
};

/* mirrored */
struct PairDPDCM {
    int ncolors;
    float a[N_COLOR], g[N_COLOR], s[N_COLOR], spow;
};

struct PairDPDLJ {
    float a, g, s, spow;
    float ljs, lje;
};

struct PairAdhesion {
    float k1, k2;
};

struct PairPa {
    float x, y, z;
    float vx, vy, vz;
    /* optional fields */
    int color;
};

struct PairFo {
    float x, y, z;
};

struct PairSFo {
    float x, y, z;
    /* stress */
    float sxx, syy, szz, sxy, sxz, syz;
};
#define _S_ static __device__
#define _I_ static __device__

_S_ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}

static const float EPS = 1e-6;
enum {NORM_OK, NORM_BIG, NORM_SMALL};
_S_ int norm(/*io*/ float3 *pos, /**/ float *pr, float *pinvr) {
    /* normalize r = [x, y, z], sets |r| and 1/|r| if not big */
    float x, y, z, invr, r, r2;
    x = pos->x; y = pos->y; z = pos->z;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return NORM_BIG;
    else if (r2 < EPS) {
        *pr = pos->x = pos->y = pos->z = 0;
        return NORM_SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        pos->x = x; pos->y = y; pos->z = z; *pr = r; *pinvr = invr;
        return NORM_OK;
    }
}

_S_ float ker_wrf(float s, float x) {
    return powf(x, s);
}

_S_ float magn_dpd(float a, float g, float s, float kpow,
                   float rnd, float r, float ev) {
    float wr, wc;
    float rm, f0;

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = ker_wrf(kpow, rm);

    f0  = (-g * wr * ev + s * rnd) * wr;
    f0 +=                        a * wc;
    return f0;
}

_S_ float force_magn(const PairDPD *p, float rnd, float ev, float r, float) {
    return magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
}

_S_ void magn2fo(float f0, float3 er, float3 dr, /**/ PairFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;
}

_S_ void magn2fo(float f0, float3 er, float3 dr, /**/ PairSFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;

    f->sxx = 0.5f * f->x * dr.x;
    f->sxy = 0.5f * f->x * dr.y;
    f->sxz = 0.5f * f->x * dr.z;
    f->syy = 0.5f * f->y * dr.y;
    f->syz = 0.5f * f->y * dr.z;
    f->szz = 0.5f * f->z * dr.z;
}

_S_ void make_zero(PairFo *f) {
    f->x = f->y = f->z = 0;
}

_S_ void make_zero(PairSFo *f) {
    f->x = f->y = f->z = 0;
    f->sxx = f->sxy = f->sxz = 0;
    f->syy = f->syz = f->szz = 0;
}

template <typename Param, typename Fo>
_I_ void pair_force(const Param *p, PairPa a, PairPa b, float rnd, /**/ Fo *f)
{
    float r, invr, ev, f0;
    float3 dr, er, dv;
    int vnstat; /* vector normalization status */

    dr.x = a.x - b.x;
    dr.y = a.y - b.y;
    dr.z = a.z - b.z;

    dv.x = a.vx - b.vx;
    dv.y = a.vy - b.vy;
    dv.z = a.vz - b.vz;

    er = dr;

    vnstat = norm(/*io*/ &er, /*o*/ &r, &invr);
    if (vnstat == NORM_BIG) {
        make_zero(f);
        return;
    }

    ev = dot<float>(&er, &dv);

    f0 = force_magn(p, rnd, ev, r, invr);
    magn2fo(f0, er, dr, /**/ f);
}

_S_ int colors2pid(int ca, int cb) {
    int c0, c1;
    c0 = ca < cb ? ca : cb;
    c1 = ca < cb ? cb : ca;
    return c1 * (c1+1) / 2 + c0;
}

template <typename Fo>
_I_ void pair_force(const PairDPDC *pc, PairPa a, PairPa b, float rnd, /**/ Fo *f) {
    PairDPD p;
    int pid;
    pid = colors2pid(a.color, b.color);
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd, /**/ f);
}

/* mirrored: parameters from particle "a" only */
template <typename Fo>
_I_ void pair_force(const PairDPDCM *pc, PairPa a, PairPa b, float rnd, /**/ Fo *f) {
    PairDPD p;
    int pid = a.color;
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd, /**/ f);
}


_I_ void pair_add(const PairFo *b, /**/ PairFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;
}

_I_ void pair_add(const PairSFo *b, /**/ PairSFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;

    a->sxx += b->sxx;
    a->sxy += b->sxy;
    a->sxz += b->sxz;
    a->syy += b->syy;
    a->syz += b->syz;
    a->szz += b->szz;
}

#undef _S_
#undef _I_
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

#ifndef UDX_DISTR_EXCH_KL_COMMON
#define UDX_DISTR_EXCH_KL_COMMON
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
#endif /* UDX_DISTR_EXCH_KL_COMMON */

namespace rnd {
/* random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
   mean = 0, variance = 1 */

// passes of logistic map
static __const__ int N = 18;
// spacing coefficints for low discrepancy numbers
static __const__ float gold   = 0.6180339887498948482;
static __const__ float silver = 0.4142135623730950488;
static __const__ float bronze = 0.00008877875787352212838853023;
static __const__ float tin    = 0.00004602357186447026756768986;
static __const__ float sqrt2 = 1.41421356237309514547;

/************************* Trunk generator ***********************
 * Make one global random number per each timestep
 * cite G. Marsaglia
 * passes BigCrush
 *****************************************************************/

/************************* Branch generator **********************
 * Make one random number per pair of particles per timestep
 * Based on the Logistic map on interval [-1,1]
 * each branch no weaker than trunk
 * zero dependency between branches
 *****************************************************************/

// floating point version of LCG
__inline__ __device__ float rem( float r ) {
    return r - floorf( r );
}

// FMA wrapper for the convenience of switching rouding modes
__inline__ __device__ float FMA( float x, float y, float z ) {
    return __fmaf_rz( x, y, z );
}

// logistic rounds
// <3> : 4 FMA + 1 MUL
// <2> : 2 FMA + 1 MUL
// <1> : 1 FMA + 1 MUL

template<int N> __inline__ __device__ float __logistic_core( float x )
{
    float x2 = x * x;
    float r = FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
    return __logistic_core < N - 2 > ( r );
}

template<> __inline__ __device__ float __logistic_core<1>( float x ) {
    return FMA( 2.0 * x, x, -1.0 );
}

template<> __inline__ __device__ float __logistic_core<0>( float x ) {
    return x;
}

inline __device__ float mean0var1ii( float seed, int u, int v )
{
    float p = rem( ( ( u & 0x3FF ) * gold ) + u * bronze + ( ( v & 0x3FF ) * silver ) + v * tin ); // safe for large u or v
    float l = __logistic_core<N>( seed - p );
    return l * sqrt2;
}

inline __device__ float mean0var1uu( float seed, uint u, uint v )
{
    // 7 FLOPS
    float p = rem( ( ( u & 0x3FFU ) * gold ) + u * bronze + ( ( v & 0x3FFU ) * silver ) + v * tin ); // safe for large u or v
    // 45+1 FLOPS
    float l = __logistic_core<N>( seed - p );
    // 1 FLOP
    return l * sqrt2;
}

}

struct PaArray_v {
    const float *pp;
};

struct PaCArray_v {
    const float *pp;
    const int *cc;   /* colors */
};
#ifndef UDX_FLU_PARRAY_COMMON
#define UDX_FLU_PARRAY_COMMON
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
#endif /* UDX_FLU_PARRAY_COMMON */
#define _S_ static __device__
#define _I_ static __device__

template<typename PA>
_S_ void parray_get_pp(PA a, int i, /**/ PairPa *p) {
    const float2 *pp = (const float2 *) a.pp;
    float2 s0, s1, s2;
    s0 = __ldg(pp + 3*i + 0);
    s1 = __ldg(pp + 3*i + 1);
    s2 = __ldg(pp + 3*i + 2);

    p->x = s0.x;
    p->y = s0.y;
    p->z = s1.x;

    p->vx = s1.y;
    p->vy = s2.x;
    p->vz = s2.y;
}

_I_ void parray_get(PaArray_v a, int i, /**/ PairPa *p) // <1>
{
    parray_get_pp(a, i, /**/ p);
}

_I_ void parray_get(PaCArray_v a, int i, /**/ PairPa *p) // <2>
{
    parray_get_pp(a, i, /**/ p);
    p->color = a.cc[i];
}

#undef _S_
#undef _I_

struct FoArray_v {
    float *ff;
};

struct FoSArray_v {
    float *ff;
    float *ss;
};
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
#define _I_ static __device__
#define _S_ static __device__

template <int S, typename Fo, typename FA>
_S_ void farray_add_force(const Fo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    af[X] += S * f->x;
    af[Y] += S * f->y;
    af[Z] += S * f->z;
}

_S_ void farray_add_stress(const PairSFo *f, int i, /**/ FoSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    s[XX] += f->sxx;
    s[XY] += f->sxy;
    s[XZ] += f->sxz;
    s[YY] += f->syy;
    s[YZ] += f->syz;
    s[ZZ] += f->szz;
}

template <int S>
_I_ void farray_add(const PairFo *f, int i, /**/ FoArray_v a) // <1>
{
    farray_add_force<S>(f, i, /**/ a);
}

template <int S>
_I_ void farray_add(const PairSFo *f, int i, /**/ FoSArray_v a) // <2>
{
    farray_add_force<S>(f, i, /**/ a);
    farray_add_stress(f, i, /**/ a);
}


template <int S, typename Fo, typename FA>
_S_ void farray_atomic_add_force(const Fo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    atomicAdd(&af[X], S * f->x);
    atomicAdd(&af[Y], S * f->y);
    atomicAdd(&af[Z], S * f->z);
}

_S_ void farray_atomic_add_stress(const PairSFo *f, int i, /**/ FoSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    atomicAdd(&s[XX], f->sxx);
    atomicAdd(&s[XY], f->sxy);
    atomicAdd(&s[XZ], f->sxz);
    atomicAdd(&s[YY], f->syy);
    atomicAdd(&s[YZ], f->syz);
    atomicAdd(&s[ZZ], f->szz);
}

template <int S>
_I_ void farray_atomic_add(const PairFo *f, int i, /**/ FoArray_v a)
{
    farray_atomic_add_force<S>(f, i, /**/ a);
}

template <int S>
_I_ void farray_atomic_add(const PairSFo *f, int i, /**/ FoSArray_v a)
{
    farray_atomic_add_force<S>(f, i, /**/ a);
    farray_atomic_add_stress(f, i, /**/ a);
}


_I_ PairFo farray_fo0(FoArray_v)
{
    PairFo f;
    f.x = f.y = f.z = 0;
    return f;
}

_I_ PairSFo farray_fo0(FoSArray_v)
{
    PairSFo f;
    f.x = f.y = f.z = 0;
    f.sxx = f.sxy = f.sxz = 0;
    f.syy = f.syz = f.szz = 0;
    return f;
}

#undef _I_
#undef _S_

#ifndef UDX_FLU_TYPE_COMMON
#define UDX_FLU_TYPE_COMMON
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
#endif /* UDX_FLU_TYPE_COMMON */

struct PairParams;
struct int3;
struct FoArray;

void fhalo_apply(const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray);

template<typename Parray>
struct LFrag_v { /* "local" fragment */
    Parray parray;
    const int *ii; /* index */
    int n;
};

template<typename Parray>
struct RFrag_v { /* "remote" fragment */
    Parray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

template <typename Parray>
using LFrag_v26 = Sarray<LFrag_v<Parray>, 26>;

template <typename Parray>
using RFrag_v26 = Sarray<RFrag_v<Parray>, 26>;
namespace fhalo_dev {
template <typename Parray>
static __device__ void assert_frag(int3 L, int i, const RFrag_v<Parray> frag) {
    int xs, ys, zs; /* sizes */
    int dx, dy, dz;
    xs = frag.xcells; ys = frag.ycells; zs = frag.zcells;
    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    assert(xs * ys * zs == frag_dev::ncell(L, i));
    assert(frag_dev::d2i(dx, dy, dz) == i);
}

template <typename Parray>
static __device__ void assert_rc(int3 L, const RFrag_v<Parray> frag, int i, int row, int col, int jump) {
    /* i: base cell id */
    int fid, nmax;
    int dx, dy, dz;
    int j;
    assert(row == 1 || row == 2 || row == 3);
    assert(col == 1 || col == 2 || col == 3);

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    assert(dx == -1 || dx == 0 || dx == 1);
    assert(dy == -1 || dy == 0 || dy == 1);
    assert(dz == -1 || dz == 0 || dz == 1);
    fid  = frag_dev::d2i(dx, dy, dz);
    nmax = frag_dev::ncell(L, fid) + 1;

    for (j = 0 ; j < row; j++) {
        assert(i       < nmax);
        assert(i + col < nmax);
        i += jump;
    }
}
struct Map { /* helps to find remote particle */
    int  org0, org1, org2;
    int cnt0, cnt1, cnt2;
};

/* true if `i' bigger than the number of remote particles */
static __device__ int endp(const Map m, int i) { return i >= m.cnt2; }
static __device__ int m2id(const Map m, int i) {
    /* return remote particle id */
    int m1, m2;
    int id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}

static __device__ int get(const int *a, int i) { return a[i]; }
static __device__ Map r2map0(const int *start, int i, int row, int col, int jump) {
    int j;
    int o[3] = {0}, c[3] = {0};
    Map m;
    assert(row == 1 || row == 2 || row == 3);
    assert(col == 1 || col == 2 || col == 3);

    for (j = 0 ; j < row; j++) {
        o[j] = get(start, i);
        c[j] = get(start, i + col) - o[j];
        i += jump;
    }
    c[1] += c[0]; c[2] += c[1];
    o[1] -= c[0]; o[2] -= c[1];

    m.org0 = o[0]; m.org1 = o[1]; m.org2 = o[2];
    m.cnt0 = c[0]; m.cnt1 = c[1]; m.cnt2 = c[2];
    return m;
}


static __device__ int facep(int dx, int dy, int dz) { /* face? */
    int x, y, z;
    x =  dx  && !dy && !dz;
    y =  !dx &&  dy && !dz;
    z =  !dx && !dy &&  dz;
    return x || y || z;
}
static __device__ void xyz2rc_face(int dx, int dy, int dz,
                                   int xc, int yc, int zc,
                                   int xs, int ys, int zs,
                                   /**/ int *prow, int *pcol, int *pjump) {
    int row, col, jump;
    if (!facep(dx, dy, dz)) {
        printf("xyz: %d %d %d\n", dx, dy, dz);
        assert(0);
    }
    if (dx) {
        row   = zs;
        col   = ys;
        jump = yc;
    } else if (dy) {
        row   = zs;
        col   = xs;
        jump = xc;
    } else { /* (dz) */
        row   = ys;
        col   = xs;
        jump = xc;
    }
    *prow = row; *pcol = col; *pjump = jump;
}

static __device__ void xyz2rc(int type,
                              int dx, int dy, int dz, /* fragment information */
                              int xc, int yc, int zc,
                              int xs, int ys, int zs, /* size */
                              /**/ int *prow, int *pcol, int *pjump) {
    int row, col, jump;
    if      (type == flu::FACE)
        xyz2rc_face(dx, dy, dz, xc, yc, zc, xs, ys, zs, /**/ &row, &col, &jump);
    else if (type == flu::EDGE) {
        row = jump = 1;
        col = max(xs, max(ys, zs));
    }
    else if (type == flu::CORNER) {
        row = col = jump = 1;
    } else {
        printf("%s:%d: illigal fragmant type: %d [%d %d %d]\n",
               __FILE__, __LINE__, type, dx, dy, dz);
        assert(0);
    }
    *prow = row; *pcol = col; *pjump = jump;
}

static __device__  void r2size(float r, int nc, int S, /**/ int *pl, int *ps) {
    int i, s, l;
    i = (int)(r + S/2);
    l = max(0,  i - 1);
    s = min(nc, i + 2) - l;
    *pl = l; *ps = s;
}

template <typename Parray>
static __device__ Map r2map(int3 L, const RFrag_v<Parray> frag, float x, float y, float z) {
    /* coordinate [r] to map */
    int id; /* base id */
    int xl, yl, zl; /* low */
    int xs, ys, zs; /* size */
    int dx, dy, dz;
    int xc, yc, zc;
    int row, col, jump;

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    xc = frag.xcells; yc = frag.ycells; zc = frag.zcells;

    id = 0; xs = 1; ys = 1; zs = 1;
    if (dz == 0) {r2size(z, zc, L.z, /**/ &zl, &zs); id += zl;}
    id *= yc;
    if (dy == 0) {r2size(y, yc, L.y, /**/ &yl, &ys); id += yl;}
    id *= xc;
    if (dx == 0) {r2size(x, xc, L.x, /**/ &xl, &xs); id += xl;}

    xyz2rc(frag.type,
           dx, dy, dz,
           xc, yc, zc,
           xs, ys, zs,
           &row, &col, &jump);
    assert_rc(L, frag, id, row, col, jump);
    return r2map0(frag.start, id, row, col, jump);
}
template <typename Farray>
struct Fo {
    Farray farray;
    int i;
};

static __device__ float random(int aid, int bid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? aid : bid;
    a2 = mask ? bid : aid;
    return rnd::mean0var1uu(seed, a1, a2);
}

template<typename Par, typename Parray, typename PFo>
static __device__ void force0(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> bfrag, const Map m, const PairPa a, int aid,
                              /**/ PFo *fa) {
    PairPa b;
    PFo f;
    int i;
    int bid; /* id of b */
    float rndval;

    for (i = 0; !endp(m, i); i ++ ) {
        bid = m2id(m, i);
        parray_get(bfrag.parray, bid, /**/ &b);
        rndval = random(aid, bid, rnd.seed, rnd.mask);
        pair_force(&params, a, b, rndval, /**/ &f);
        pair_add(&f, /**/ fa);
    }
}

template<typename Par, typename Parray, typename Farray>
static __device__ void force1(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> frag, const Map m, const PairPa p, int id, Fo<Farray> f) {
    auto fa = farray_fo0(f.farray);
    force0(params, rnd, frag, m, p, id, /**/ &fa);
    farray_atomic_add<1>(&fa, f.i, f.farray);
}

template<typename Par, typename Parray, typename Farray>
static __device__ void force2(Par params, int3 L, const RFrag_v<Parray> frag, const flu::RndFrag rnd, PairPa p, int id, /**/ Fo<Farray> f) {
    Map m;
    m = r2map(L, frag, p.x, p.y, p.z);
    p.x -= frag.dx * L.x;
    p.y -= frag.dy * L.y;
    p.z -= frag.dz * L.z;

    force1(params, rnd, frag, m, p, id, /**/ f);
}

template <typename Farray>
static __device__ Fo<Farray> i2f(const int *ii, Farray farray, int i) {
    /* local id and index to force */
    Fo<Farray> f;
    f.i = ii[i];
    f.farray = farray;
    return f;
}

template <typename Parray, typename Farray>
static __device__ Fo<Farray> Lfrag2f(const LFrag_v<Parray> frag, Farray farray, int i) {
    return i2f(frag.ii, farray, i);
}

template <typename Par, typename Parray, typename Farray>
static __device__ void force3(Par params, int3 L, const LFrag_v<Parray> afrag, const RFrag_v<Parray> bfrag, const flu::RndFrag rnd, int i,
                              /**/ Farray farray) {
    PairPa p;
    Fo<Farray> f;
    parray_get(afrag.parray, i, &p);
    f = Lfrag2f(afrag, farray, i);
    force2(params, L, bfrag, rnd, p, i, f);
}

template <typename Par, typename Parray, typename Farray>
__global__ void apply(int fid, Par params, int3 L, const LFrag_v<Parray> lfrag, const RFrag_v<Parray> rfrag, const flu::RndFrag rnd,
                      /**/ Farray farray) {
    int i; /* particle id */

    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= lfrag.n) return;

    force3(params, L, lfrag, rfrag, rnd, i, /**/ farray);
}
}
enum {
    PADDING = 16 /* warpsize / 2 */
};

template <typename Parray>
static void get_view_lfrag(const flu::LFrag26 f, LFrag_v26<Parray> *v) {
    Parray pav;
    LFrag_v<Parray> *vi;
    const flu::LFrag *fi;

    for (int i = 0; i < 26; ++i) {
        vi = &v->d[i];
        fi = &f.d[i];
        parray_get_view(&fi->parray, &pav);

        vi->parray = pav;
        vi->ii = fi->ii;
        vi->n  = fi->n;
    }
}

template <typename Parray>
static void get_view_rfrag(const flu::RFrag26 f, RFrag_v26<Parray> *v) {
    Parray pav;
    RFrag_v<Parray> *vi;
    const flu::RFrag *fi;

    for (int i = 0; i < 26; ++i) {
        vi = &v->d[i];
        fi = &f.d[i];
        parray_get_view(&fi->parray, &pav);

        vi->parray = pav;
        vi->start = fi->start;
        vi->dx = fi->dx;
        vi->dy = fi->dy;
        vi->dz = fi->dz;
        vi->xcells = fi->xcells;
        vi->ycells = fi->ycells;
        vi->zcells = fi->zcells;
        vi->type = fi->type;
    }
}

static bool is_colored(const flu::RFrag26 f) {
    return parray_is_colored(&(f.d[0].parray));
}

template <typename Par, typename Parray, typename Farray>
static void interactions(Par params, int3 L, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ Farray farray) {
    int n, i;
    for (i = 0; i < 26; ++i) {
        n = lfrags.d[i].n;
        KL(fhalo_dev::apply, (k_cnf(n)), (i, params, L, lfrags.d[i], rfrags.d[i], rrnd.d[i], /**/ farray));
    }
}

template <typename Par, typename Parray>
static void apply(Par params, int3 L, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {

    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, lfrags, rfrags, rrnd, /**/ farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, lfrags, rfrags, rrnd, /**/ farray_v);
    }
}

static void apply_grey(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    PairDPD pv;
    LFrag_v26<PaArray_v> lfragsv;
    RFrag_v26<PaArray_v> rfragsv;

    pair_get_view_dpd(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);

    apply(pv, L, lfragsv, rfragsv, rrnd, /**/ farray);
}

static void apply_color(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    PairDPDC pv;
    LFrag_v26<PaCArray_v> lfragsv;
    RFrag_v26<PaCArray_v> rfragsv;

    pair_get_view_dpd_color(params, &pv);
    get_view_lfrag(lfrags, &lfragsv);
    get_view_rfrag(rfrags, &rfragsv);

    apply(pv, L, lfragsv, rfragsv, rrnd, /**/ farray);
}

void fhalo_apply(const PairParams *params, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ const FoArray *farray) {
    if (is_colored(rfrags))
        apply_color(params, L, lfrags, rfrags, rrnd, /**/ farray);
    else
        apply_grey(params, L, lfrags, rfrags, rrnd, /**/ farray);
}

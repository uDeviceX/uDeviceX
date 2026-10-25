// generated from udx.flat.cu
#include "udx.config.cuh"
#include "udx.device-api-common.cuh"

/* begin src/exch/map/imp.cu */
#include <stdlib.h>
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

/* begin src/exch/map/type.h */
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
/* begin src/exch/map/imp.h */
// tag::int[]
void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map);               // <1>
void emap_fin(int nfrags, EMap *map);                                       // <2>
void emap_reini(int nw, int nfrags, /**/ EMap map);                         // <3>
void emap_scan(int nw, int nfrags, /**/ EMap map);                          // <4>
void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]); // <5>
void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int tot_counts[], int *counts[]);
// end::int[]
/* end src/exch/map/imp.h */
/* begin src/algo/utils/shfl.h */
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
/* end src/algo/utils/shfl.h */
/* begin src/exch/map/dev/main.h */
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
/* end src/exch/map/dev/main.h */

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
/* end src/exch/map/imp.cu */

#line 1 "src/exch/mesh/imp.cu"

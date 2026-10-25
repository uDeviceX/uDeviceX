// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/algo/scan/imp.cu */
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
#ifndef UDX_ALGO_KL_COMMON
#define UDX_ALGO_KL_COMMON
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
/* begin src/d/api.h */
#ifndef UDX_ALGO_DEVICE_API_COMMON
#define UDX_ALGO_DEVICE_API_COMMON
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

/* begin src/algo/scan/imp.h */
struct Scan;

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
// end::interface[]
/* end src/algo/scan/imp.h */

namespace scan_dev {
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
/* begin src/algo/scan/dev.h */
__global__ void compress(const int nentries, const int4 *const in, uchar4 *const out) {
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (4 * gid >= nentries)
        return;

    const int4 entry = in[gid];

    out[gid] = make_uchar4(entry.x, entry.y, entry.z, entry.w);
}

template <int NWARP>
__global__ void breduce(const uint4 *vin, unsigned int *vout, int n) {

    const int wid = threadIdx.x/32;
    const int lid = threadIdx.x%32;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    uint4 val = make_uint4(0,0,0,0);

    __shared__ unsigned int shtmp[NWARP];

    if (tid < n) val = vin[tid];

    val.x = ((val.x & 0xFF000000) >> 24) + ((val.x & 0xFF0000) >> 16) + ((val.x & 0xFF00) >> 8) + (val.x & 0xFF);
    val.y = ((val.y & 0xFF000000) >> 24) + ((val.y & 0xFF0000) >> 16) + ((val.y & 0xFF00) >> 8) + (val.y & 0xFF);
    val.z = ((val.z & 0xFF000000) >> 24) + ((val.z & 0xFF0000) >> 16) + ((val.z & 0xFF00) >> 8) + (val.z & 0xFF);
    val.w = ((val.w & 0xFF000000) >> 24) + ((val.w & 0xFF0000) >> 16) + ((val.w & 0xFF00) >> 8) + (val.w & 0xFF);
    val.x += val.y + val.z + val.w;

#pragma unroll
    for(int i = 16; i > 0; i >>= 1)
        val.x += shfl_down((int)val.x, i);

    if (0 == lid)
        shtmp[wid] = val.x;

    __syncthreads();
    if (0 == wid) {
        val.x = (lid < NWARP) ? shtmp[lid] : 0;

#pragma unroll
        for(int i = 16; i > 0; i >>= 1)
            val.x += shfl_down((int)val.x, i);
    }
    if (0 == threadIdx.x) vout[blockIdx.x] = val.x;
    return;
}

template <int BDIM>
__global__ void bexscan(unsigned int *v, int n) {

    extern __shared__ unsigned int shtmp[];

    for(int i = threadIdx.x; i < n; i += BDIM) shtmp[i] = v[i];

    int i = threadIdx.x;
    __syncthreads();
    for(; n >= BDIM; i += BDIM, n -= BDIM) {

        __syncthreads();
        if (i > 0 && 0 == threadIdx.x)
            shtmp[i] += shtmp[i-1];

        unsigned int a = 0;

#pragma unroll
        for(int j = 1; j < BDIM; j <<= 1) {
            a = 0;

            __syncthreads();
            if (threadIdx.x >= j) a = shtmp[i] + shtmp[i-j];

            __syncthreads();
            if (threadIdx.x >= j) shtmp[i] = a;
        }
        v[i] = shtmp[i];
    }
    if (threadIdx.x < n) {

        __syncthreads();
        if (i > 0 && 0 == threadIdx.x)
            shtmp[i] += shtmp[i-1];

        unsigned int a = 0;
        for(int j = 1; j < BDIM; j <<= 1) {
            a = 0;

            __syncthreads();
            if (threadIdx.x >= j) a = shtmp[i] + shtmp[i-j];

            __syncthreads();
            if (threadIdx.x >= j) shtmp[i] = a;
        }
        v[i] = shtmp[i];
    }
    return;
}

template <int NWARP>
__global__ void gexscan(const uint4 *vin, unsigned int *offs, uint4 *vout, int n) {

    const int wid = threadIdx.x/32;
    const int lid = threadIdx.x%32;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    uint4 val[4];

    __shared__ unsigned int woff[NWARP];

    if (tid < n) val[0] = vin[tid];
    else         val[0] = make_uint4(0,0,0,0);

    // read bytes B0 B1 B2 B3 into VAL[*].y VAL[*].z VAL[*].w VAL[*].x
    val[3].y = (val[0].w & 0x000000FF);
    val[3].z = (val[0].w & 0x0000FF00) >>  8;
    val[3].w = (val[0].w & 0x00FF0000) >> 16;
    val[3].x = (val[0].w & 0xFF000000) >> 24;

    val[2].y = (val[0].z & 0x000000FF);
    val[2].z = (val[0].z & 0x0000FF00) >>  8;
    val[2].w = (val[0].z & 0x00FF0000) >> 16;
    val[2].x = (val[0].z & 0xFF000000) >> 24;

    val[1].y = (val[0].y & 0x000000FF);
    val[1].z = (val[0].y & 0x0000FF00) >>  8;
    val[1].w = (val[0].y & 0x00FF0000) >> 16;
    val[1].x = (val[0].y & 0xFF000000) >> 24;

    val[0].y = (val[0].x & 0x000000FF);
    val[0].z = (val[0].x & 0x0000FF00) >>  8;
    val[0].w = (val[0].x & 0x00FF0000) >> 16;
    val[0].x = (val[0].x & 0xFF000000) >> 24;

    uint4 tu4;
    tu4.x = val[0].x + val[0].y + val[0].z + val[0].w;
    tu4.y = val[1].x + val[1].y + val[1].z + val[1].w;
    tu4.z = val[2].x + val[2].y + val[2].z + val[2].w;
    tu4.w = val[3].x + val[3].y + val[3].z + val[3].w;

    unsigned int tmp = tu4.x + tu4.y + tu4.z + tu4.w;

    tu4.w = tmp;
#pragma unroll
    for(int i = 1; i < 32; i <<= 1)
        tu4.w += (lid >= i)*shfl_up((int)tu4.w, i);

    if (lid == 31)
        if (wid < NWARP-1) woff[wid+1] = tu4.w;
        else               woff[0]     = (blockIdx.x > 0) ? offs[blockIdx.x-1] : 0;

    tu4.w -= tmp;

    __syncthreads();
    if (0 == wid) {
        tmp = (lid < NWARP) ? woff[lid] : 0;
#pragma unroll
        for(int i = 1; i < NWARP; i <<= 1)
            tmp += (lid >= i)*shfl_up((int)tmp, i);

        if (lid < NWARP) woff[lid] = tmp;
    }
    __syncthreads();

    if (tid >= n) return;

    uint4 lps;
    lps.x = woff[wid] + tu4.w;
    lps.y = lps.x + tu4.x;
    lps.z = lps.y + tu4.y;
    lps.w = lps.z + tu4.z;

    val[0].x = lps.x;
    val[1].x = lps.y;
    val[2].x = lps.z;
    val[3].x = lps.w;

    val[0].y += val[0].x;
    val[1].y += val[1].x;
    val[2].y += val[2].x;
    val[3].y += val[3].x;

    val[0].z += val[0].y;
    val[1].z += val[1].y;
    val[2].z += val[2].y;
    val[3].z += val[3].y;

    val[0].w += val[0].z;
    val[1].w += val[1].z;
    val[2].w += val[2].z;
    val[3].w += val[3].z;

    vout += tid*4;

#pragma unroll
    for(int i = 0; i < 4; i++)
        vout[i] = val[i];
    return;
}
/* end src/algo/scan/dev.h */
}

#if   defined(DEV_CUDA)
/* begin src/algo/scan/cuda/type.h */
struct Scan {
    int size;
    unsigned int  *tmp;
    unsigned char *compressed;
};
/* end src/algo/scan/cuda/type.h */
/* begin src/algo/scan/cuda/imp.h */
enum { THREADS = 128 };
static void scan0(const unsigned char *input, int size, /**/ uint *output, /*w*/ uint *tmp) {
    int nblocks = ((size / 16) + THREADS - 1 ) / THREADS;
    KL(scan_dev::breduce<THREADS/32>, (nblocks, THREADS                ), ((uint4 *)input, tmp, size / 16));
    KL(scan_dev::bexscan<THREADS>   , (1, THREADS, nblocks*sizeof(uint)), (tmp, nblocks));
    KL(scan_dev::gexscan<THREADS/32>, (nblocks, THREADS                ), ((uint4 *)input, tmp, (uint4 *)output, size / 16));
}

void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan *w) {
    if (size > w->size)
        ERR(" size = %d > w->size = %d", size, w->size);
    if (!d::is_device_pointer(input))  ERR("`input`  is not a device pointer");
    if (!d::is_device_pointer(output)) ERR("`output` is not a device pointer");

    KL(scan_dev::compress, (k_cnf(size)), (size, (const int4*) input, /**/ (uchar4 *) w->compressed));
    scan0(w->compressed, size, /**/ (uint*) output, /*w*/ w->tmp);
}

void scan_ini(int size, /**/ Scan **work) {
    Scan *w;
    EMALLOC(1, work);
    w = *work;
    w->size = size;
    Dalloc(&w->tmp, 64 * 64 * 64 / THREADS);
    Dalloc(&w->compressed, 4 * size);
}

void scan_fin(/**/ Scan *w) {
    Dfree(w->tmp);
    Dfree(w->compressed);
    EFREE(w);
}
/* end src/algo/scan/cuda/imp.h */
#elif defined(DEV_CPU)
/* begin src/algo/scan/cpu/type.h */
struct Scan { };
/* end src/algo/scan/cpu/type.h */
/* begin src/algo/scan/cpu/imp.h */
void scan_ini(int, Scan**) { }
void scan_fin(Scan*) { }

void scan_apply(const int* input, int size, /**/ int* output, /*w*/ Scan*) {
    int i, s;
    s = 0;
    for (i = 0; i < size; i++) {
        output[i] = s;
        s += input[i];
    }

}
/* end src/algo/scan/cpu/imp.h */
#else
  #error DEV_* is undefined
#endif
/* end src/algo/scan/imp.cu */

#line 1 "src/algo/vectors/imp.cpp"

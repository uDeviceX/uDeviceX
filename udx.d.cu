// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/d/api.cu */
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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

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

namespace d {
/* begin src/d/common.h */
static const char *emsg0; /* last error message */
const char *emsg() { return emsg0; }
/* end src/d/common.h */

#if   defined(DEV_CUDA)
/* begin src/d/cuda/imp.h */
#define SZ(a) (int)(sizeof(a)/sizeof(*(a)))
static enum cudaMemcpyKind k2k0[] = {  /* [k]ind to [k]ind */
    cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault };
static enum cudaMemcpyKind k2k(int k) {
    if (k < 0 || k >= SZ(k2k0)) ERR("unknown kind: %d", k);
    return k2k0[k];
}

static int R(cudaError_t e) {
    if (e == cudaSuccess) return 0;
    else {
        emsg0 = cudaGetErrorString(e);
        return 1;
    }
}

int is_device_pointer(const void *p) {
    cudaPointerAttributes a;
    cudaError_t e;
    e = cudaPointerGetAttributes(&a, p);
    if      (e == cudaErrorInvalidValue) return 0;
    else if (a.type != cudaMemoryTypeDevice) return 0;
    else return 1;
}

int GetSymbolAddress(void **devPtr, const void *symbol) {
    return R(cudaGetSymbolAddress(devPtr, symbol));
}

int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, int kind0) {
    enum cudaMemcpyKind kind;
    kind = k2k(kind0);
    return R(cudaMemcpyToSymbol(symbol, src, count, offset, kind));
}

int MemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, int kind0) {
    enum cudaMemcpyKind kind;
    kind = k2k(kind0);
    return R(cudaMemcpyFromSymbol(dst, symbol, count, offset, kind));
}

int HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    return R(cudaHostGetDevicePointer (pDevice, pHost, flags));
}

int Memcpy (void *dst, const void *src, size_t count, int kind0) {
    enum cudaMemcpyKind kind = k2k(kind0);
    return R(cudaMemcpy(dst, src, count, kind));
}

int MemsetAsync (void *devPtr, int value, size_t count, Stream_t stream) {
    if (stream != 0) ERR("streams are not supported");
    return R(cudaMemsetAsync(devPtr, value, count));
}

int Memset (void *devPtr, int value, size_t count) {
    return R(cudaMemset(devPtr, value, count));
}

int MemcpyAsync (void *dst, const void *src, size_t count, int kind0, Stream_t stream) {
    enum cudaMemcpyKind kind;
    if (stream != 0) ERR("streams are not supported");
    switch (kind0) {
    case MemcpyDeviceToDevice:
        if (!is_device_pointer(dst)) ERR("`dst` is not a device pointer");
        if (!is_device_pointer(src)) ERR("`src` is not a device pointer");
        break;
    }
    kind = k2k(kind0);
    return R(cudaMemcpyAsync (dst, src, count, kind));
}

int DeviceSynchronize (void) {
    return R(cudaDeviceSynchronize());
}

int PeekAtLastError(void) {
    return R(cudaPeekAtLastError());
}
/* end src/d/cuda/imp.h */
#  if  defined(CUDA_ALLOC_RELEASE)
/* begin src/d/cuda/release/alloc.h */
int alloc_pinned(void **pHost, size_t size) {
    return R(cudaHostAlloc(pHost, size, cudaHostAllocMapped));
}

int Malloc(void **devPtr, size_t size) {
    return R(cudaMalloc(devPtr, size));
}

int Free (void *devPtr) {
    return R(cudaFree (devPtr));
}

int FreeHost (void *hstPtr) {
    return R(cudaFreeHost(hstPtr));
}
/* end src/d/cuda/release/alloc.h */
#  elif defined(CUDA_ALLOC_DEBUG)
/* begin src/d/cuda/debug/alloc.h */
/* for sysconf */
#include <unistd.h>

static size_t tot_d = 0;
static size_t tot_h = 0;

static double b2mb(size_t byte) { return (double) byte / (double) (1 << 20); }

static size_t memory() {
    size_t p, sz;
    p  = sysconf(_SC_PHYS_PAGES);
    sz = sysconf(_SC_PAGE_SIZE);
    return p * sz;
}

int alloc_pinned(void **pHost, size_t size) {
    cudaError_t status;
    tot_h += size;
    msg_print("[alloc_pinned] %g MB (total %g MB)", b2mb(size), b2mb(tot_h));
    status = cudaHostAlloc(pHost, size, cudaHostAllocMapped);
    msg_print("host memory: %g MB", b2mb(memory()));
    return R(status);
}

int Malloc(void **devPtr, size_t size) {
    cudaError_t status;
    tot_d += size;
    msg_print("[Malloc] %g MB (total %g MB)", b2mb(size), b2mb(tot_d));
    status = cudaMalloc(devPtr, size);
    msg_print("host memory: %g MB", b2mb(memory()));
    return R(status);
}

int Free (void *devPtr) {
    msg_print("[Free]");
    return R(cudaFree (devPtr));
}

int FreeHost (void *hstPtr) {
    msg_print("[FreeHost]");
    return R(cudaFreeHost(hstPtr));
}
/* end src/d/cuda/debug/alloc.h */
#  else
#    error CUDA_ALLOC_* is undefined
#  endif
#elif defined(DEV_CPU)
/* begin src/d/cpu/imp.h */
/* [v]oid add : shift by `n' number of bytes */
#define vadd(p, n) (void*)((char*)(p) + (n))

int alloc_pinned(void **pHost, size_t size) {
    UC(emalloc(size, pHost));
    return 0;
}

int is_device_pointer(const void *ptr) { return 1; }

int Malloc(void **p, size_t size) {
    UC(emalloc(size, p));
    return 0;
}

int GetSymbolAddress(void **, const void*) {
    ERR("GetSymbolAddress cannot be called if devices is empulated by CPU");
    return 0;
}

int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, int /*kind*/) {
    memcpy(vadd(symbol, offset), src, count);
    return 0;
}

int MemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, int /*kind*/) {
    memcpy(dst, vadd(symbol, offset), count);
    return 0;
}

int HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    *pDevice = pHost;
    return 0;
}

int Memcpy (void *dst, const void *src, size_t count, int /*kind*/) {
    memcpy(dst, src, count);
    return 0;
}

int MemsetAsync (void *devPtr, int value, size_t count, Stream_t) {
    memset(devPtr, value, count);
    return 0;
}

int Memset (void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
    return 0;
}

int MemcpyAsync (void * dst, const void * src, size_t count, int /*kind*/, Stream_t) {
    memcpy(dst, src, count);
    return 0;
}

int Free (void *devPtr) {
    free(devPtr);
    return 0;
}

int FreeHost (void *hstPtr) {
    free(hstPtr);
    return 0;
}

int DeviceSynchronize (void) {
    return 0;
}

int PeekAtLastError(void) {
    return 0;
}
/* end src/d/cpu/imp.h */
#else
#  error DEV_* is undefined
#endif

}
/* end src/d/api.cu */

#line 1 "src/dbg/conf.cpp"

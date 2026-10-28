#include "config.cuh"
#include "device.cuh"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void emalloc(size_t, void **);
void msg_print(const char *fmt, ...);

void error_stack_push(const char *file, int line);
void error_stack_pop();
void error_signal(const char *file, int line, const char *fmt, ...);
void error_report();
void error_abort();
bool error_get();

#define ERR(fmt, ...) do { \
    error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
    error_report(); \
    error_abort(); \
} while (0)

#define UC(F) do { \
    error_stack_push(__FILE__, __LINE__); \
    F; \
    if (error_get()) { \
        error_report(); \
        error_abort(); \
    } \
    error_stack_pop(); \
} while (0)

namespace d {
static const char *emsg0;
const char *emsg() { return emsg0; }

#if   defined(DEV_CUDA)
#define SZ(a) (int)(sizeof(a)/sizeof(*(a)))
static enum cudaMemcpyKind k2k0[] = {
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
#  if  defined(CUDA_ALLOC_RELEASE)
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
#  elif defined(CUDA_ALLOC_DEBUG)

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
#  else
#    error CUDA_ALLOC_* is undefined
#  endif
#elif defined(DEV_CPU)

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

int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, int         ) {
    memcpy(vadd(symbol, offset), src, count);
    return 0;
}

int MemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, int         ) {
    memcpy(dst, vadd(symbol, offset), count);
    return 0;
}

int HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    *pDevice = pHost;
    return 0;
}

int Memcpy (void *dst, const void *src, size_t count, int         ) {
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

int MemcpyAsync (void * dst, const void * src, size_t count, int         , Stream_t) {
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
#else
#  error DEV_* is undefined
#endif

}

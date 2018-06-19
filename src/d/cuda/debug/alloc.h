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

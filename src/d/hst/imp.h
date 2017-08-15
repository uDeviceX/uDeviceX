#define miss(n) MSG(n ": missing")

void ini() { }

cudaError_t Malloc(void **p, size_t size) {
    *p = malloc(size);
    return cudaSuccess;
}

cudaError_t MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    miss("MemcpyToSymbol");
    return cudaSuccess;
}

cudaError_t HostAlloc (void **pHost, size_t size, unsigned int flags) {
    miss("HostAlloc");
    return cudaSuccess;
}

cudaError_t HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    miss("HostGetDevicePointer");
    return cudaSuccess;
}

cudaError_t Memcpy (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    miss("Memcpy");
    return cudaSuccess;    
}

cudaError_t MemsetAsync (void *devPtr, int value, size_t count, cudaStream_t stream) {
    miss("MemsetAsync");
    return cudaSuccess;
}

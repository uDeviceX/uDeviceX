#define miss(n) MSG(n ": missing")
/* [v]oid add */
#define vadd(p, n) (void*)((char*)(p) + (n))

void ini() { }

cudaError_t Malloc(void **p, size_t size) {
    *p = malloc(size);
    return cudaSuccess;
}

cudaError_t MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind) {
    memcpy(vadd(symbol, offset), src, count);
    return cudaSuccess;
}

cudaError_t HostAlloc (void **pHost, size_t size, unsigned int) {
    *pHost = malloc(size);
    return cudaSuccess;
}

cudaError_t HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    *pDevice = pHost;
    return cudaSuccess;
}

cudaError_t Memcpy (void *dst, const void *src, size_t count, enum cudaMemcpyKind) {
    memcpy(dst, src, count);
    return cudaSuccess;    
}

cudaError_t MemsetAsync (void *devPtr, int value, size_t count, cudaStream_t stream) {
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t Memset (void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t MemcpyAsync (void * dst, const void * src, size_t count, enum cudaMemcpyKind, cudaStream_t) {
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t Free (void *devPtr) {
    free(devPtr);
    return cudaSuccess;
}

cudaError_t DeviceSynchronize (void) {
    return cudaSuccess;
}

const char *GetErrorString (cudaError_t error) {
    static const char e[] = "api: cpu: error\n";
    return e;
}

void ini() {
    /* panda specific for multi-gpu testing
       int device = m::rank % 2 ? 0 : 2; */
    int device = 0;
    CC(cudaSetDevice(device));
}

cudaError_t Malloc(void **devPtr, size_t size) {
    return cudaMalloc(devPtr, size);
}

cudaError_t MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t HostAlloc(void **pHost, size_t size, unsigned int flags) {
    return cudaHostAlloc(pHost, size, flags);
}

cudaError_t HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    return cudaHostGetDevicePointer (pDevice, pHost, flags);
}

cudaError_t Memcpy (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t MemsetAsync (void *devPtr, int value, size_t count, cudaStream_t stream) {
    return cudaMemsetAsync(devPtr, value, count, stream);
}

cudaError_t MemcpyAsync (void * dst, const void * src, size_t count, enum cudaMemcpyKind
                         kind, cudaStream_t stream) {
    return cudaMemcpyAsync (dst, src, count, kind, stream);
}

cudaError_t Free (void *devPtr) {
    return cudaFree (devPtr);
}

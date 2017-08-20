static int R(cudaError_t e) {
    if (e == cudaSuccess) return 0;
    else {
        emsg0 = cudaGetErrorString(e);
        return 1;
    }
}

int ini() {
    /* panda specific for multi-gpu testing
       int device = m::rank % 2 ? 0 : 2; */
    int device = 0;
    return R(cudaSetDevice(device));
}

int Malloc(void **devPtr, size_t size) {
    return R(cudaMalloc(devPtr, size));
}

int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    return R(cudaMemcpyToSymbol(symbol, src, count, offset, kind));
}

int HostAlloc(void **pHost, size_t size, unsigned int flags) {
    return R(cudaHostAlloc(pHost, size, flags));
}

int HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    return R(cudaHostGetDevicePointer (pDevice, pHost, flags));
}

int Memcpy (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    return R(cudaMemcpy(dst, src, count, kind));
}

int MemsetAsync (void *devPtr, int value, size_t count, cudaStream_t stream) {
    return R(cudaMemsetAsync(devPtr, value, count, stream));
}

int Memset (void *devPtr, int value, size_t count) {
    return R(cudaMemset(devPtr, value, count));
}

int MemcpyAsync (void * dst, const void * src, size_t count, enum cudaMemcpyKind
                         kind, cudaStream_t stream) {
    return R(cudaMemcpyAsync (dst, src, count, kind, stream));
}

int Free (void *devPtr) {
    return R(cudaFree (devPtr));
}

int DeviceSynchronize (void) {
    return R(cudaDeviceSynchronize());
}

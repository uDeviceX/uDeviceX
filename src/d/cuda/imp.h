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

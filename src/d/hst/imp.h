void ini() { }

cudaError_t Malloc(void **p, size_t size) {
    *p = malloc(size);
    return cudaSuccess;
}

cudaError_t MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset,
                           enum cudaMemcpyKind kind) {
    MSG("MemcpyToSymbol is not implemented on host");
    return cudaSuccess;
}

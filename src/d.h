namespace d {  /* a wrapper for device API */
void ini();
cudaError_t Malloc(void **p, size_t);
cudaError_t MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset=0,
                           enum cudaMemcpyKind kind=cudaMemcpyHostToDevice);
cudaError_t HostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t HostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
cudaError_t Memcpy (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
}

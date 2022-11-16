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

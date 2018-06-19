static size_t b2mb(size_t byte) { return byte / 1000000; }

int alloc_pinned(void **pHost, size_t size) {
    msg_print("[alloc_pinned] %ld MB", b2mb(size));
    return R(cudaHostAlloc(pHost, size, cudaHostAllocMapped));
}

int Malloc(void **devPtr, size_t size) {
    msg_print("[Malloc] %ld MB", b2mb(size));
    return R(cudaMalloc(devPtr, size));
}

int Free (void *devPtr) {
    msg_print("[Free]");
    return R(cudaFree (devPtr));
}

int FreeHost (void *hstPtr) {
    msg_print("[FreeHost]");
    return R(cudaFreeHost(hstPtr));
}

/* [v]oid add : shift by `n' number of bytes */
#define vadd(p, n) (void*)((char*)(p) + (n))

int ini() { return 0; }

int Malloc(void **p, size_t size) {
    *p = malloc(size);
    return 0;
}

int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, int /*kind*/) {
    memcpy(vadd(symbol, offset), src, count);
    return 0;
}

int HostAlloc (void **pHost, size_t size, unsigned int) {
    *pHost = malloc(size);
    return 0;
}

int HostGetDevicePointer (void **pDevice, void *pHost, unsigned int flags) {
    *pDevice = pHost;
    return 0;
}

int Memcpy (void *dst, const void *src, size_t count, int /*kind*/) {
    memcpy(dst, src, count);
    return 0;    
}

int MemsetAsync (void *devPtr, int value, size_t count, Stream_t) {
    memset(devPtr, value, count);
    return 0;
}

int Memset (void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
    return 0;
}

int MemcpyAsync (void * dst, const void * src, size_t count, int /*kind*/, Stream_t) {
    memcpy(dst, src, count);
    return 0;
}

int Free (void *devPtr) {
    free(devPtr);
    return 0;
}

int FreeHost (void *hstPtr) {
    free(hstPtr);
    return 0;
}

int DeviceSynchronize (void) {
    return 0;
}

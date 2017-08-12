namespace cc {
inline void check(cudaError_t rc, const char *file, int line) {
    if (rc != cudaSuccess) {
        MSG("%s:%d: %s\n", file, line, cudaGetErrorString(rc));
        abort();
    }
}
}

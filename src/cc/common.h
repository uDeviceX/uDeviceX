namespace cc {
inline void check(cudaError_t rc, const char *file, int line) {
    if (rc != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s:%d\n", cudaGetErrorString(rc), file, line);
        abort();
    }
}
}

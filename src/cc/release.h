/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cudaAssert((ans), __FILE__, __LINE__);} while (0)

inline void cudaAssert(cudaError_t rc, const char *file, int line) {
    if (rc != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(rc), file, line);
        abort();
    }
}

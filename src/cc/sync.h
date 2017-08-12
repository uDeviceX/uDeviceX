/* Aggressive syncronization */
#define CC(ans)                                             \
    do { cudaDeviceSynchronize();                           \
        cudaAssert((ans), __FILE__, __LINE__);              \
        cudaDeviceSynchronize(); } while (0)

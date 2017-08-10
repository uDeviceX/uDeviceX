/* Aggressive syncronization */

/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cudaDeviceSynchronize();                           \
        cudaAssert((ans), __FILE__, __LINE__);              \
        cudaDeviceSynchronize(); } while (0)

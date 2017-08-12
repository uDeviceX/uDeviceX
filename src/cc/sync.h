/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        cudaDeviceSynchronize();                            \
        cc::check((ans), __FILE__, __LINE__);               \
        cudaDeviceSynchronize();                            \
    } while (0)

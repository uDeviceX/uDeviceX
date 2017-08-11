/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cudaAssert((ans), __FILE__, __LINE__);} while (0)

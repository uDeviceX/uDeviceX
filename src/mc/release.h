/* [m]pi [c]heck */
#define MC(ans)                                             \
    do { mpiAssert((ans), __FILE__, __LINE__); } while (0)

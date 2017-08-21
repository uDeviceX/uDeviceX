/* [m]pi [c]heck */
#define MC(ans)                                             \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

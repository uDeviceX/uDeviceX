/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)

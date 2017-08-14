/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do {                                                    \
        MSG("cc: %s", #ans);                                \
        cc::check((ans), __FILE__, __LINE__);               \
    } while (0)

/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do {                                                    \
        MSG("cc: %s:%d: %s", __FILE__, __LINE__, #ans);     \
        cc::check((ans), __FILE__, __LINE__);               \
    } while (0)

/* [c]cuda [c]heck */
#define CC(ans)                                                 \
    do {                                                        \
        msg_print("cc: %s:%d: %s", __FILE__, __LINE__, #ans);   \
        cc::check((ans), __FILE__, __LINE__);                   \
        d::PeekAtLastError();                                   \
    } while (0)

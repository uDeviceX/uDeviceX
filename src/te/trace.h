#define TE(te, D, n)                                            \
    do {                                                        \
        MSG("te: %s:%d: %s: n = %d", __FILE__, __LINE__, #D);   \
        TE_CALL(te, D, n);                                      \
    } while (0)

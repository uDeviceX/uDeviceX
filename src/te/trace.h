#define TE(te, D, n)                            \
    do {                                        \
        MSG("te: %s:%d:", __FILE__, __LINE__);  \
        TE_CALL(te, D, n);                      \
    } while (0)

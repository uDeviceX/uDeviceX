#define TE(te, D, n)                            \
    do {                                        \
        msg_print("te: %s:%d: %s: n = %d",      \
                  __FILE__, __LINE__, #D, n);   \
        TE_CALL(te, D, n);                      \
    } while (0)

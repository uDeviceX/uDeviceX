#define TE(te, D, n)                                     \
    do {                                                 \
        MSG("te: %s:%d: n = %d", __FILE__, __LINE__, n); \
        TE_CALL(te, D, n);                               \
    } while (0)

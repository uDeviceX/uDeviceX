namespace dbg {
#define ESC(...) __VA_ARGS__
#define DBG(F, A, M)                            \
    do {                                        \
        DBG_BEFORE(#F, M);                      \
        DBG_CALL(F, A);                         \
        DBG_AFTER(#F, M);                       \
    } while(0)
} // dbg

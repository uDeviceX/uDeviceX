namespace dbg {
#define ESC(...) __VA_ARGS__
#define DBG(Fun, A, F, L, M)                    \
    do {                                        \
        DBG_BEFORE(#Fun, F, L, M);              \
        DBG_CALL(Fun, A);                       \
        DBG_AFTER(#Fun, F, L, M);               \
    } while(0)
} // dbg

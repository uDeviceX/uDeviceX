namespace dbg {
#define DBG_BEFORE(s, F, L, M) printf("%s:%d: DBG: %s : %s\n", F, L, s, M);
#define DBG_CALL(F, A) F A;
#define DBG_AFTER(s, F, L, M)
} // dbg

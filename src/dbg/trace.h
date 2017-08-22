namespace dbg {
#define DBG_BEFORE(s, M) printf("DBG: %s : %s\n", s, M);
#define DBG_CALL(F, A) F A;
#define DBG_AFTER(s)
} // dbg
